# DocMind 前端 E2E 与视觉回归

该目录包含两层互补的 macOS Chromium 门禁：

- 默认套件覆盖注册后自动登录、独立登录、流式文档问答、文档上传、Skills 调用与 artifact 下载，以及异步评估的 `pending → running → completed/failed` 状态流转。它在浏览器层拦截 API 和 SSE，并返回固定数据，适合快速、确定性的 UI 契约与视觉回归。
- `playwright.fullstack.config.js` 启动真实 FastAPI，再由 FastAPI 托管真实前端。它使用临时 SQLite、Qdrant 本地存储和进程内任务执行器，不连接 PostgreSQL、Redis、外部 Qdrant、Celery 或模型服务；覆盖真实 `/health`、注册、JWT 鉴权、基础 API 以及页面重载后的登录态恢复。

## 本地运行

```bash
cd frontend
npm ci
npx playwright install chromium
npm run test:e2e:ci
npm run test:e2e:fullstack
```

全栈测试默认使用仓库根目录的 `.venv/bin/python`；可通过 `DOCMIND_E2E_PYTHON` 指向其他已安装 `requirements.txt` 的 Python。启动器会为每次运行创建独立临时目录，并在退出时清理数据库、上传目录、Skill 工作区和 Qdrant 数据。还可用 `DOCMIND_E2E_FULLSTACK_PORT` 覆盖默认端口 `4180`。

UI/视觉 HTML 报告位于 `frontend/playwright-report/`，全栈报告位于 `frontend/playwright-report-fullstack/`，失败截图、视频、trace 和 JUnit 结果位于 `frontend/test-results/`。这些目录不会进入 Git。

## 视觉基线

版本化基线位于 `e2e/visual-baselines/chromium/*.png.base64`。运行时会把文本形式的基线解码到 `test-results/` 后交给 Playwright 像素比较；这样既保留真实 PNG 比对，也让基线可通过普通文本补丁审查。

仅在确认 UI 变化符合预期后更新基线：

```bash
npm run test:e2e:update
```

该命令先在 `test-results/visual-baselines/` 生成 PNG。审核图片后，将 PNG 的 Base64 内容更新到对应的 `.png.base64` 文件，再重新运行 `npm run test:e2e:ci`。v3.2 版本化基线与 CI 都固定使用 macOS Chromium，并仅保留 0.1% 抗锯齿余量。套件还会主动把登录卡背景改成洋红色，并要求同一个 `toHaveScreenshot` 断言确实拒绝该变化，防止视觉门禁退化成永远通过。Linux 基线和 Windows 浏览器不在本次验收范围内。

## CI

`.github/workflows/frontend-e2e.yml` 在主分支推送、PR 和手动触发时运行。macOS runner 安装锁定的 Node 依赖和 Python `requirements.txt`，先运行 mock/视觉套件，再运行真实 FastAPI 全栈套件。CI 使用单 worker、失败重试，并始终上传两套 HTML 报告、JUnit、截图、视频和 trace，便于定位偶发或视觉问题。

仓库主 CI 还会执行 Ruff 严重错误规则、模型/Schema 的 mypy 契约检查、所有前端与桌面构建脚本的 JavaScript 语法检查，以及桌面 Rust 的 `fmt`、`clippy` 和单元测试。

Playwright 是本子任务唯一新增的前端依赖；项目原先没有浏览器测试运行器，而自动 E2E 与视觉回归目标明确要求真实浏览器及截图比较能力。
