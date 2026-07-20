use std::{
    collections::HashMap,
    env, fs,
    io::{self, Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use serde::Serialize;
use tauri::{AppHandle, Manager, State};

const BACKEND_HOST: &str = "127.0.0.1";
const BACKEND_PORT: u16 = 8000;
const SIDECAR_NAME: &str = "docmind-sidecar";
const TARGET_TRIPLE: &str = env!("DOCMIND_TARGET_TRIPLE");
const DESKTOP_ENV_TEMPLATE: &str = include_str!("../../sidecar/default.env");
const SIDECAR_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

#[cfg(unix)]
fn request_child_shutdown(child: &mut Child) -> io::Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }

    // SAFETY: POSIX kill(2) does not dereference pointers. The child id comes
    // from std::process::Child and SIGTERM (15) is defined by POSIX/macOS.
    extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
    }
    let result = unsafe { kill(child.id() as i32, 15) };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
fn request_child_shutdown(child: &mut Child) -> io::Result<()> {
    // std has no cross-platform graceful console signal. Windows packaging is
    // outside v2.2 validation; retain the existing force-stop fallback there.
    child.kill()
}

fn terminate_child(child: &mut Child, timeout: Duration) -> bool {
    if child.try_wait().ok().flatten().is_some() {
        return true;
    }
    if request_child_shutdown(child).is_ok() {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(_)) => return true,
                Ok(None) => thread::sleep(Duration::from_millis(100)),
                Err(_) => break,
            }
        }
    }

    let _ = child.kill();
    let _ = child.wait();
    false
}

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
        self.set_status("starting", "正在准备 DocMind 本地数据目录…");
        if let Err(message) = self.launch(&app) {
            self.fail(&message);
        }
    }

    fn launch(&self, app: &AppHandle) -> Result<(), String> {
        if api_is_healthy() {
            return Err(
                "127.0.0.1:8000 已有服务占用。请先停止另一个 DocMind 实例，再重试。".to_owned(),
            );
        }

        let runtime_dir = app
            .path()
            .app_config_dir()
            .map_err(|error| format!("无法定位桌面配置目录：{error}"))?;
        let data_dir = runtime_dir.join("data");
        let log_dir = runtime_dir.join("logs");
        prepare_runtime_directories(&data_dir, &log_dir)?;
        let config_file = ensure_config_file(&runtime_dir)?;
        let sidecar = locate_sidecar(app)?;
        let environment = runtime_environment(&config_file, &data_dir);

        self.set_status("starting", "正在启动内置 DocMind 服务…");
        let mut child = spawn_sidecar(
            &sidecar,
            &config_file,
            &data_dir,
            &runtime_dir,
            &environment,
            &log_dir.join("sidecar.log"),
        )?;

        self.set_status("starting", "正在迁移本地数据并检查服务…");
        if let Err(message) = wait_for_api(&mut child) {
            terminate_child(&mut child, SIDECAR_SHUTDOWN_TIMEOUT);
            return Err(format!(
                "{message} 详细信息：{}",
                log_dir.join("sidecar.log").display()
            ));
        }

        self.children
            .lock()
            .map_err(|_| "无法记录内置服务进程。".to_owned())?
            .push(child);
        self.set_status("ready", "DocMind 本地服务已就绪。");
        Ok(())
    }

    fn stop_children(&self) {
        let Ok(mut children) = self.children.lock() else {
            return;
        };
        for child in children.iter_mut() {
            terminate_child(child, SIDECAR_SHUTDOWN_TIMEOUT);
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
fn backend_origin() -> String {
    format!("http://{BACKEND_HOST}:{BACKEND_PORT}")
}

#[tauri::command]
fn retry_backend(app: AppHandle, manager: State<'_, Arc<BackendManager>>) {
    manager.inner().start(app);
}

fn prepare_runtime_directories(data_dir: &Path, log_dir: &Path) -> Result<(), String> {
    for directory in [
        data_dir.to_path_buf(),
        data_dir.join("uploads"),
        data_dir.join("skill_workspaces"),
        data_dir.join("qdrant"),
        data_dir.join("secrets"),
        log_dir.to_path_buf(),
    ] {
        fs::create_dir_all(&directory)
            .map_err(|error| format!("无法创建 {}：{error}", directory.display()))?;
    }
    Ok(())
}

fn ensure_config_file(runtime_dir: &Path) -> Result<PathBuf, String> {
    let config_file = runtime_dir.join("desktop.env");
    if !config_file.exists() {
        fs::write(&config_file, DESKTOP_ENV_TEMPLATE)
            .map_err(|error| format!("无法创建桌面配置 {}：{error}", config_file.display()))?;
    }
    Ok(config_file)
}

fn sqlite_database_url(database_path: &Path) -> String {
    let normalized = database_path.to_string_lossy().replace('\\', "/");
    format!("sqlite+aiosqlite:///{normalized}")
}

fn runtime_environment(env_file: &Path, data_dir: &Path) -> HashMap<String, String> {
    let mut environment = HashMap::new();
    environment.insert(
        "DOCMIND_ENV_FILE".to_owned(),
        env_file.to_string_lossy().into_owned(),
    );
    environment.insert(
        "DOCMIND_DATA_DIR".to_owned(),
        data_dir.to_string_lossy().into_owned(),
    );
    environment.insert(
        "DATABASE_URL".to_owned(),
        sqlite_database_url(&data_dir.join("docmind.db")),
    );
    environment.insert(
        "QDRANT_PATH".to_owned(),
        data_dir.join("qdrant").to_string_lossy().into_owned(),
    );
    environment.insert("TASK_EXECUTION_MODE".to_owned(), "local".to_owned());
    environment.insert("AGENT_RUN_LOCK_BACKEND".to_owned(), "sqlite".to_owned());
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
    environment.insert("PYTHONUTF8".to_owned(), "1".to_owned());
    environment.insert("PYTHONUNBUFFERED".to_owned(), "1".to_owned());
    environment.insert("NO_PROXY".to_owned(), "127.0.0.1,localhost".to_owned());
    environment.insert(
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY".to_owned(),
        "YES".to_owned(),
    );
    environment
}

fn executable_suffix() -> &'static str {
    if cfg!(target_os = "windows") {
        ".exe"
    } else {
        ""
    }
}

fn bundled_sidecar_name() -> String {
    format!("{SIDECAR_NAME}{}", executable_suffix())
}

fn build_sidecar_name() -> String {
    format!("{SIDECAR_NAME}-{TARGET_TRIPLE}{}", executable_suffix())
}

fn locate_sidecar(app: &AppHandle) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Some(configured) = env::var_os("DOCMIND_SIDECAR_PATH") {
        candidates.push(PathBuf::from(configured));
    }
    if let Ok(current_executable) = env::current_exe() {
        if let Some(directory) = current_executable.parent() {
            candidates.push(directory.join(bundled_sidecar_name()));
            candidates.push(directory.join(build_sidecar_name()));
        }
    }
    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join(bundled_sidecar_name()));
        candidates.push(resource_dir.join("binaries").join(bundled_sidecar_name()));
        candidates.push(resource_dir.join("binaries").join(build_sidecar_name()));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("binaries")
            .join(build_sidecar_name()),
    );

    if let Some(found) = candidates.iter().find(|candidate| candidate.is_file()) {
        return found
            .canonicalize()
            .map_err(|error| format!("无法解析内置服务路径：{error}"));
    }

    let searched = candidates
        .iter()
        .map(|candidate| candidate.display().to_string())
        .collect::<Vec<_>>()
        .join("；");
    Err(format!(
        "安装包缺少 {SIDECAR_NAME}。请重新安装完整的 DocMind 安装包。已检查：{searched}"
    ))
}

fn spawn_sidecar(
    program: &Path,
    config_file: &Path,
    data_dir: &Path,
    current_dir: &Path,
    environment: &HashMap<String, String>,
    log_path: &Path,
) -> Result<Child, String> {
    let log = fs::File::create(log_path)
        .map_err(|error| format!("无法创建日志文件 {}：{error}", log_path.display()))?;
    let backend_port = BACKEND_PORT.to_string();
    let parent_pid = std::process::id().to_string();
    let data_dir = data_dir.to_string_lossy().into_owned();
    let config_file = config_file.to_string_lossy().into_owned();
    let mut command = Command::new(program);
    command
        .args([
            "serve",
            "--host",
            BACKEND_HOST,
            "--port",
            backend_port.as_str(),
            "--parent-pid",
            parent_pid.as_str(),
            "--data-dir",
            data_dir.as_str(),
            "--config",
            config_file.as_str(),
        ])
        .current_dir(current_dir)
        .envs(environment)
        .stdin(Stdio::null())
        .stdout(Stdio::from(
            log.try_clone().map_err(|error| error.to_string())?,
        ))
        .stderr(Stdio::from(log));

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    command
        .spawn()
        .map_err(|error| format!("无法启动内置服务 {}：{error}", program.display()))
}

fn wait_for_api(child: &mut Child) -> Result<(), String> {
    for _ in 0..120 {
        if api_is_healthy() {
            return Ok(());
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("无法检查内置服务状态：{error}"))?
        {
            return Err(format!("内置服务在启动期间退出（{status}）。"));
        }
        thread::sleep(Duration::from_secs(1));
    }
    Err("DocMind 内置服务在 120 秒内未就绪。".to_owned())
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
        .is_ok_and(|_| response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200"))
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
        .invoke_handler(tauri::generate_handler![
            backend_status,
            backend_origin,
            retry_backend
        ])
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
    fn desktop_runtime_environment_selects_embedded_services() {
        let environment = runtime_environment(
            Path::new("/tmp/docmind/desktop.env"),
            Path::new("/tmp/docmind/data"),
        );

        assert_eq!(
            environment.get("DATABASE_URL"),
            Some(&"sqlite+aiosqlite:////tmp/docmind/data/docmind.db".to_owned())
        );
        assert_eq!(
            environment.get("QDRANT_PATH"),
            Some(&"/tmp/docmind/data/qdrant".to_owned())
        );
        assert_eq!(
            environment.get("TASK_EXECUTION_MODE"),
            Some(&"local".to_owned())
        );
        assert_eq!(environment.get("DOCMIND_DESKTOP"), Some(&"1".to_owned()));
        assert_eq!(
            environment.get("AGENT_RUN_LOCK_BACKEND"),
            Some(&"sqlite".to_owned())
        );
        assert!(!environment.contains_key("REDIS_URL"));
    }

    #[test]
    fn managed_backend_starts_from_a_known_status() {
        let status = BackendStatus::default();
        assert_eq!(status.state, "starting");
        assert!(!status.message.is_empty());
    }

    #[test]
    fn frontend_origin_uses_the_managed_sidecar_address() {
        assert_eq!(backend_origin(), "http://127.0.0.1:8000");
    }

    #[test]
    fn sidecar_names_cover_build_and_installed_layouts() {
        assert_eq!(
            bundled_sidecar_name(),
            format!("docmind-sidecar{}", executable_suffix())
        );
        assert!(build_sidecar_name().contains(TARGET_TRIPLE));
    }

    #[test]
    fn desktop_config_does_not_expose_runtime_storage_switches() {
        assert!(!DESKTOP_ENV_TEMPLATE.contains("DATABASE_URL"));
        assert!(!DESKTOP_ENV_TEMPLATE.contains("REDIS_URL"));
        assert!(!DESKTOP_ENV_TEMPLATE.contains("QDRANT_PATH"));
        assert!(!DESKTOP_ENV_TEMPLATE.contains("TASK_EXECUTION_MODE"));
    }

    #[cfg(unix)]
    #[test]
    fn sidecar_shutdown_requests_sigterm_before_force_kill() {
        let mut child = Command::new("/bin/sh")
            .args(["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn SIGTERM-aware child");
        thread::sleep(Duration::from_millis(100));

        assert!(terminate_child(&mut child, Duration::from_secs(3)));
        assert!(child.try_wait().expect("read child status").is_some());
    }
}
