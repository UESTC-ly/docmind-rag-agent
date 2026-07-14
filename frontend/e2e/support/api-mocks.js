const timestamp = "2026-07-13T08:00:00Z";

function fulfillJson(route, body, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json; charset=utf-8",
    body: JSON.stringify(body),
  });
}

export async function installApiMocks(page, options = {}) {
  const state = {
    requests: [],
    uploaded: false,
    evaluationPollCount: 0,
    email: "reader@example.com",
  };
  const evaluation = options.evaluation || {};

  await page.route("**/auth/login", async (route) => {
    const body = route.request().postDataJSON();
    state.email = body.email;
    state.requests.push({ path: "/auth/login", body });
    await fulfillJson(route, { access_token: "e2e-token", token_type: "bearer" });
  });
  await page.route("**/auth/register", async (route) => {
    const body = route.request().postDataJSON();
    state.email = body.email;
    state.requests.push({ path: "/auth/register", body });
    await fulfillJson(route, { id: 7, email: body.email }, 201);
  });
  await page.route("**/auth/me", (route) => fulfillJson(route, { id: 7, email: state.email }));
  await page.route("**/chat/conversations", (route) =>
    fulfillJson(route, [
      { id: 12, title: "项目知识库", created_at: timestamp, updated_at: timestamp },
    ])
  );
  await page.route(/\/chat\/conversations\/\d+$/, (route) => fulfillJson(route, []));
  await page.route("**/chat/stream", async (route) => {
    state.requests.push({ path: "/chat/stream", body: route.request().postDataJSON() });
    const body = [
      "event: meta",
      `data: ${JSON.stringify({
        conversation_id: 23,
        sources: [
          {
            document_id: 5,
            chunk_index: 2,
            score: 0.927,
            content: "DocMind 使用混合检索与可插拔技能回答文档问题。",
          },
        ],
      })}`,
      "",
      "event: token",
      `data: ${JSON.stringify({ text: "DocMind 会结合文档证据回答问题。" })}`,
      "",
      "",
    ].join("\n");
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream; charset=utf-8",
      headers: { "Cache-Control": "no-cache" },
      body,
    });
  });

  await page.route("**/documents/upload", async (route) => {
    state.requests.push({ path: "/documents/upload" });
    state.uploaded = true;
    await fulfillJson(route, {
      id: 8,
      filename: "architecture.md",
      status: "pending",
      chunk_count: 0,
      created_at: timestamp,
    }, 201);
  });
  await page.route("**/documents/", (route) =>
    fulfillJson(
      route,
      state.uploaded
        ? [
            {
              id: 8,
              filename: "architecture.md",
              status: "completed",
              chunk_count: 4,
              created_at: timestamp,
            },
          ]
        : []
    )
  );
  await page.route(/\/documents\/\d+$/, (route) => fulfillJson(route, null, 204));

  await page.route("**/agent/skills", (route) =>
    fulfillJson(route, [
      {
        name: "search_knowledge_base",
        description: "检索已上传文档",
        available: true,
        execution_mode: "python",
        grounding_mode: "hybrid_rag",
        produces_download: false,
        package: null,
      },
    ])
  );
  await page.route("**/agent/chat", (route) => {
    state.requests.push({ path: "/agent/chat", body: route.request().postDataJSON() });
    return fulfillJson(route, options.agentResult || {
      conversation_id: 24,
      answer: "已完成知识库检索。",
      trace: [],
      artifacts: [],
    });
  });

  await page.route("**/eval/datasets", (route) =>
    fulfillJson(route, evaluation.datasets || [])
  );
  await page.route(/\/eval\/runs(?:\/\d+)?$/, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    state.requests.push({ path, method: request.method() });

    if (request.method() === "POST") {
      await fulfillJson(route, evaluation.createdRun || {
        id: 71,
        dataset_id: 31,
        status: "pending",
        created_at: timestamp,
        completed_at: null,
      }, 202);
      return;
    }

    const responses = evaluation.pollResponses || [];
    const response = responses[Math.min(state.evaluationPollCount, Math.max(0, responses.length - 1))];
    state.evaluationPollCount += 1;
    if (evaluation.pollDelayMs) {
      await new Promise((resolve) => setTimeout(resolve, evaluation.pollDelayMs));
    }
    await fulfillJson(route, response || evaluation.createdRun || {
      id: 71,
      dataset_id: 31,
      status: "pending",
      created_at: timestamp,
      completed_at: null,
    });
  });

  return state;
}
