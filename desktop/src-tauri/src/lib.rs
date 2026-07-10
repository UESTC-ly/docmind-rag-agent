use std::{
    collections::HashMap,
    env,
    ffi::OsString,
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use serde::Serialize;
use tauri::{AppHandle, Manager, State};

const BACKEND_PORT: u16 = 8000;

#[derive(Clone, Serialize)]
struct BackendStatus {
    state: String,
    message: String,
}

impl Default for BackendStatus {
    fn default() -> Self {
        Self {
            state: "starting".to_owned(),
            message: "正在准备 DocMind 本地服务…".to_owned(),
        }
    }
}

#[derive(Default)]
struct BackendManager {
    status: Mutex<BackendStatus>,
    children: Mutex<Vec<Child>>,
    launch_lock: Mutex<()>,
}

impl BackendManager {
    fn start(self: &Arc<Self>, app: AppHandle) {
        let manager = Arc::clone(self);
        thread::spawn(move || manager.run(app));
    }

    fn run(&self, app: AppHandle) {
        let _launch_guard = match self.launch_lock.lock() {
            Ok(guard) => guard,
            Err(_) => return self.fail("桌面服务启动锁不可用。"),
        };

        if self.current_status().state == "ready" {
            return;
        }

        self.stop_children();
        self.set_status("starting", "正在准备 DocMind 本地运行环境…");

        if let Err(message) = self.launch(&app) {
            self.fail(&message);
        }
    }

    fn launch(&self, app: &AppHandle) -> Result<(), String> {
        if api_is_healthy() {
            return Err(
                "127.0.0.1:8000 已有正在运行的 DocMind 服务。请先停止网页启动脚本启动的服务，再打开桌面 App。"
                    .to_owned(),
            );
        }

        let backend_root = backend_root(app)?;
        let runtime_dir = app
            .path()
            .app_config_dir()
            .map_err(|error| format!("无法创建桌面配置目录：{error}"))?;
        let data_dir = runtime_dir.join("data");
        let log_dir = runtime_dir.join("logs");
        fs::create_dir_all(data_dir.join("uploads"))
            .map_err(|error| format!("无法创建上传目录：{error}"))?;
        fs::create_dir_all(data_dir.join("skill_workspaces"))
            .map_err(|error| format!("无法创建技能工作区：{error}"))?;
        fs::create_dir_all(&log_dir).map_err(|error| format!("无法创建日志目录：{error}"))?;

        let env_file = ensure_env_file(&backend_root, &runtime_dir)?;
        let environment = runtime_environment(&env_file, &data_dir);

        self.set_status("starting", "正在启动 PostgreSQL、Redis 和 Qdrant…");
        ensure_docker()?;
        start_compose(&backend_root)?;

        self.set_status("starting", "正在检查 Python 运行环境…");
        let python = ensure_python_environment(&backend_root, &runtime_dir)?;

        self.set_status("starting", "正在启动文档解析后台任务…");
        let celery = spawn_process(
            &python,
            [
                "-m",
                "celery",
                "-A",
                "app.celery_app",
                "worker",
                "--loglevel=info",
                "--pool=solo",
            ],
            &backend_root,
            &environment,
            &log_dir.join("celery.log"),
        )?;
        self.children
            .lock()
            .map_err(|_| "无法记录 Celery 子进程。".to_owned())?
            .push(celery);

        self.set_status("starting", "正在启动 DocMind API…");
        let api = spawn_process(
            &python,
            [
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            &backend_root,
            &environment,
            &log_dir.join("api.log"),
        )?;
        self.children
            .lock()
            .map_err(|_| "无法记录 API 子进程。".to_owned())?
            .push(api);

        self.set_status("starting", "正在等待 DocMind API 就绪…");
        wait_for_api()?;
        self.set_status("ready", "DocMind 本地服务已就绪。");
        Ok(())
    }

    fn stop_children(&self) {
        let Ok(mut children) = self.children.lock() else {
            return;
        };
        for child in children.iter_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        children.clear();
    }

    fn current_status(&self) -> BackendStatus {
        self.status
            .lock()
            .map(|status| status.clone())
            .unwrap_or_else(|_| BackendStatus {
                state: "error".to_owned(),
                message: "桌面服务状态不可用。".to_owned(),
            })
    }

    fn set_status(&self, state: &str, message: &str) {
        if let Ok(mut status) = self.status.lock() {
            *status = BackendStatus {
                state: state.to_owned(),
                message: message.to_owned(),
            };
        }
    }

    fn fail(&self, message: &str) {
        self.stop_children();
        self.set_status("error", message);
    }
}

#[tauri::command]
fn backend_status(manager: State<'_, Arc<BackendManager>>) -> BackendStatus {
    manager.current_status()
}

#[tauri::command]
fn retry_backend(app: AppHandle, manager: State<'_, Arc<BackendManager>>) {
    manager.inner().start(app);
}

fn backend_root(app: &AppHandle) -> Result<PathBuf, String> {
    let development_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if development_root.join("app").is_dir() {
        return development_root
            .canonicalize()
            .map_err(|error| format!("无法定位开发后端目录：{error}"));
    }

    app.path()
        .resource_dir()
        .map(|path| path.join("backend"))
        .map_err(|error| format!("无法定位打包后的后端资源：{error}"))
}

fn ensure_env_file(backend_root: &Path, runtime_dir: &Path) -> Result<PathBuf, String> {
    let env_file = runtime_dir.join(".env");
    if env_file.exists() {
        return Ok(env_file);
    }

    let example = backend_root.join(".env.example");
    fs::copy(&example, &env_file)
        .map_err(|error| format!("首次启动时无法创建配置文件 {}：{error}", env_file.display()))?;
    Ok(env_file)
}

fn runtime_environment(env_file: &Path, data_dir: &Path) -> HashMap<String, String> {
    let mut environment = HashMap::new();
    environment.insert(
        "PATH".to_owned(),
        desktop_command_path().to_string_lossy().into_owned(),
    );
    environment.insert(
        "DOCMIND_ENV_FILE".to_owned(),
        env_file.to_string_lossy().into_owned(),
    );
    environment.insert(
        "UPLOAD_DIR".to_owned(),
        data_dir.join("uploads").to_string_lossy().into_owned(),
    );
    environment.insert(
        "SKILL_WORKSPACE_DIR".to_owned(),
        data_dir
            .join("skill_workspaces")
            .to_string_lossy()
            .into_owned(),
    );
    environment.insert("DOCMIND_DESKTOP".to_owned(), "1".to_owned());
    environment.insert(
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY".to_owned(),
        "YES".to_owned(),
    );
    environment
}

fn ensure_docker() -> Result<(), String> {
    if command_succeeds("docker", ["info"], None) {
        return Ok(());
    }

    if cfg!(target_os = "macos") && command_succeeds("colima", ["version"], None) {
        run_command("colima", ["start"], None, None)?;
    }

    if command_succeeds("docker", ["info"], None) {
        Ok(())
    } else {
        Err(
            "Docker 未运行。请启动 Docker Desktop（macOS 也可安装并启用 Colima）后重试。"
                .to_owned(),
        )
    }
}

fn start_compose(backend_root: &Path) -> Result<(), String> {
    let compose_file = backend_root.join("docker-compose.yml");
    let compose_file = compose_file.to_string_lossy().into_owned();
    if command_succeeds("docker", ["compose", "version"], Some(backend_root)) {
        run_command(
            "docker",
            ["compose", "-f", compose_file.as_str(), "up", "-d"],
            Some(backend_root),
            None,
        )
    } else if command_succeeds("docker-compose", ["version"], Some(backend_root)) {
        run_command(
            "docker-compose",
            ["-f", compose_file.as_str(), "up", "-d"],
            Some(backend_root),
            None,
        )
    } else {
        Err("未找到 Docker Compose。请安装 Docker Compose plugin 后重试。".to_owned())
    }
}

fn ensure_python_environment(backend_root: &Path, runtime_dir: &Path) -> Result<PathBuf, String> {
    let environment_dir = runtime_dir.join("python-env");
    let python = venv_python(&environment_dir);
    if !python.exists() {
        run_command(
            "uv",
            [
                "venv",
                "--python",
                "3.12",
                environment_dir.to_string_lossy().as_ref(),
            ],
            Some(backend_root),
            None,
        )?;
    }
    run_command(
        "uv",
        [
            "pip",
            "install",
            "--python",
            python.to_string_lossy().as_ref(),
            "-r",
            backend_root
                .join("requirements.txt")
                .to_string_lossy()
                .as_ref(),
        ],
        Some(backend_root),
        None,
    )?;
    Ok(python)
}

fn venv_python(environment_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        environment_dir.join("Scripts").join("python.exe")
    } else {
        environment_dir.join("bin").join("python")
    }
}

fn spawn_process<I, S>(
    program: &Path,
    arguments: I,
    current_dir: &Path,
    environment: &HashMap<String, String>,
    log_path: &Path,
) -> Result<Child, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let log = fs::File::create(log_path)
        .map_err(|error| format!("无法创建日志文件 {}：{error}", log_path.display()))?;
    Command::new(program)
        .args(arguments.into_iter().map(|item| item.as_ref().to_owned()))
        .current_dir(current_dir)
        .envs(environment)
        .stdout(Stdio::from(
            log.try_clone().map_err(|error| error.to_string())?,
        ))
        .stderr(Stdio::from(log))
        .spawn()
        .map_err(|error| format!("无法启动 {}：{error}", program.display()))
}

fn wait_for_api() -> Result<(), String> {
    for _ in 0..90 {
        if api_is_healthy() {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(1));
    }
    Err("DocMind API 在 90 秒内未就绪。请查看桌面配置目录 logs/api.log。".to_owned())
}

fn api_is_healthy() -> bool {
    let address = SocketAddr::from(([127, 0, 0, 1], BACKEND_PORT));
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_secs(1)) else {
        return false;
    };
    if stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .is_ok_and(|_| response.starts_with("HTTP/1.1 200"))
}

fn command_succeeds<I, S>(program: &str, arguments: I, current_dir: Option<&Path>) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut command = desktop_command(program);
    command.args(arguments.into_iter().map(|item| item.as_ref().to_owned()));
    if let Some(path) = current_dir {
        command.current_dir(path);
    }
    command
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn run_command<I, S>(
    program: &str,
    arguments: I,
    current_dir: Option<&Path>,
    environment: Option<&HashMap<String, String>>,
) -> Result<(), String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut command = desktop_command(program);
    command.args(arguments.into_iter().map(|item| item.as_ref().to_owned()));
    if let Some(path) = current_dir {
        command.current_dir(path);
    }
    if let Some(values) = environment {
        command.envs(values);
    }
    command
        .status()
        .map_err(|error| format!("无法执行 {program}：{error}"))?
        .success()
        .then_some(())
        .ok_or_else(|| format!("命令执行失败：{program}"))
}

fn desktop_command(program: &str) -> Command {
    let mut command = Command::new(program);
    command.env("PATH", desktop_command_path());
    command
}

fn desktop_command_path() -> OsString {
    let mut paths: Vec<PathBuf> = env::var_os("PATH")
        .map(|value| env::split_paths(&value).collect())
        .unwrap_or_default();

    for path in [
        PathBuf::from("/opt/homebrew/bin"),
        PathBuf::from("/usr/local/bin"),
        PathBuf::from("/usr/bin"),
        PathBuf::from("/bin"),
        PathBuf::from("/usr/sbin"),
        PathBuf::from("/sbin"),
    ] {
        if !paths.contains(&path) {
            paths.push(path);
        }
    }

    if let Some(home) = env::var_os("HOME").map(PathBuf::from) {
        for path in [home.join(".local/bin"), home.join(".cargo/bin")] {
            if !paths.contains(&path) {
                paths.push(path);
            }
        }
    }

    env::join_paths(paths).unwrap_or_else(|_| OsString::from("/usr/local/bin:/usr/bin:/bin"))
}

pub fn run() {
    let backend_manager = Arc::new(BackendManager::default());
    let startup_manager = Arc::clone(&backend_manager);
    let shutdown_manager = Arc::clone(&backend_manager);

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_notification::init())
        .manage(backend_manager)
        .invoke_handler(tauri::generate_handler![backend_status, retry_backend])
        .setup(move |app| {
            startup_manager.start(app.handle().clone());
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(
                event,
                tauri::WindowEvent::CloseRequested { .. } | tauri::WindowEvent::Destroyed
            ) {
                let manager = window.state::<Arc<BackendManager>>();
                manager.stop_children();
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building DocMind desktop application");

    app.run(move |_, event| {
        if matches!(
            event,
            tauri::RunEvent::Exit | tauri::RunEvent::ExitRequested { .. }
        ) {
            shutdown_manager.stop_children();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_runtime_environment_uses_external_writable_directories() {
        let environment = runtime_environment(
            Path::new("/tmp/docmind/.env"),
            Path::new("/tmp/docmind/data"),
        );

        assert_eq!(
            environment.get("DOCMIND_ENV_FILE"),
            Some(&"/tmp/docmind/.env".to_owned())
        );
        assert_eq!(environment.get("DOCMIND_DESKTOP"), Some(&"1".to_owned()));
        assert!(environment["UPLOAD_DIR"].ends_with("data/uploads"));
        assert!(environment["SKILL_WORKSPACE_DIR"].ends_with("data/skill_workspaces"));
    }

    #[test]
    fn managed_backend_starts_from_a_known_status() {
        let status = BackendStatus::default();

        assert_eq!(status.state, "starting");
        assert!(!status.message.is_empty());
    }

    #[test]
    fn desktop_path_includes_homebrew_and_system_commands() {
        let paths: Vec<PathBuf> = env::split_paths(&desktop_command_path()).collect();

        assert!(paths.contains(&PathBuf::from("/opt/homebrew/bin")));
        assert!(paths.contains(&PathBuf::from("/usr/local/bin")));
        assert!(paths.contains(&PathBuf::from("/usr/bin")));
    }
}
