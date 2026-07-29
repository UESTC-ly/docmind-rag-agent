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
            citation_id: "D5:C2",
            score: 0.927,
            content: "DocMind 使用混合检索与可插拔技能回答文档问题。",
          },
        ],
      })}`,
      "",
      "event: token",
      `data: ${JSON.stringify({ text: "DocMind 会结合文档证据回答问题。" })}`,
      "",
      "event: verification",
      `data: ${JSON.stringify({
        contract: "citation_presence_v1",
        semantic_entailment_checked: false,
        claim_count: 1,
        supported_claim_count: 0,
        unsupported_claim_count: 1,
        citation_count: 0,
        valid_citation_count: 0,
        citation_precision: 1,
        citation_recall: 0,
        unsupported_claim_rate: 1,
        passed: false,
        claims: [{
          text: "DocMind 会结合文档证据回答问题。",
          citations: [],
          valid_citations: [],
          invalid_citations: [],
          supported: false,
          lexical_overlap: 0,
        }],
      })}`,
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
    if (options.agentError) {
      return fulfillJson(
        route,
        { detail: options.agentError.detail || "Agent request failed" },
        options.agentError.status,
      );
    }
    return fulfillJson(route, options.agentResult || {
      conversation_id: 24,
      run_id: "run-completed",
      thread_id: "run-completed",
      status: "completed",
      answer: "已完成知识库检索。",
      trace: [],
      artifacts: [],
      approval: null,
    });
  });
  await page.route("**/agent/resume", (route) => {
    state.requests.push({ path: "/agent/resume", body: route.request().postDataJSON() });
    return fulfillJson(route, options.resumeResult || {
      conversation_id: 24,
      run_id: "run-approved",
      thread_id: "run-approved",
      status: "completed",
      answer: "审批后已完成。",
      trace: [],
      artifacts: [],
      approval: null,
    });
  });
  await page.route(/\/agent\/runs\/[^/]+$/, (route) => {
    const path = new URL(route.request().url()).pathname;
    state.requests.push({ path });
    if (options.agentRunError) {
      return fulfillJson(
        route,
        { detail: options.agentRunError.detail || "Agent run unavailable" },
        options.agentRunError.status,
      );
    }
    return fulfillJson(route, options.agentRunResult || {
      conversation_id: 24,
      run_id: path.split("/").at(-1),
      thread_id: path.split("/").at(-1),
      status: "completed",
      answer: "Agent run 已完成。",
      trace: [],
      artifacts: [],
      approval: null,
    });
  });
  await page.route(/\/agent\/runs\/[^/]+\/recover$/, (route) => {
    const path = new URL(route.request().url()).pathname;
    state.requests.push({ path, method: route.request().method() });
    return fulfillJson(route, options.recoverResult || {
      conversation_id: 24,
      run_id: path.split("/").at(-2),
      thread_id: path.split("/").at(-2),
      status: "completed",
      recoverable: false,
      answer: "已从 checkpoint 恢复。",
      trace: [],
      artifacts: [],
      approval: null,
    });
  });

  await page.route("**/eval/datasets", (route) =>
    fulfillJson(route, evaluation.datasets || [])
  );
  await page.route("**/eval/pipelines", (route) =>
    fulfillJson(route, evaluation.pipelines || [
      {
        id: "configured",
        label: "当前配置",
        description: "当前配置",
        retriever: "hybrid",
        fusion: "rrf",
        reranker: "local",
        context_builder: "evidence",
        top_k: 5,
        fingerprint: "configured-fingerprint",
        spec: {},
      },
      {
        id: "dense",
        label: "Dense 基线",
        description: "Dense",
        retriever: "dense",
        fusion: "dense",
        reranker: "off",
        context_builder: "evidence",
        top_k: 5,
        fingerprint: "dense-fingerprint",
        spec: {},
      },
    ])
  );
  await page.route("**/eval/experiments", async (route) => {
    const body = route.request().postDataJSON();
    state.requests.push({ path: "/eval/experiments", body });
    await fulfillJson(route, evaluation.createdRuns || []);
  });
  await page.route(/\/eval\/runs\/\d+\/metrics$/, (route) =>
    fulfillJson(route, evaluation.metricRows || [])
  );
  await page.route(/\/eval\/runs\/\d+\/badcases(?:\?.*)?$/, (route) => {
    const runId = Number(new URL(route.request().url()).pathname.split("/").at(-2));
    return fulfillJson(
      route,
      runId === evaluation.createdRerun?.id
        ? evaluation.rerunBadcases || []
        : evaluation.badcases || [],
    );
  });
  await page.route(/\/eval\/runs\/\d+\/badcase-diff$/, (route) =>
    fulfillJson(route, evaluation.badcaseDiff || {
      baseline_run_id: 0,
      candidate_run_id: 0,
      newly_introduced: [],
      fixed: [],
      persistent: [],
      unchanged_passed_count: 0,
    })
  );
  await page.route(/\/eval\/runs\/\d+\/regression$/, (route) =>
    fulfillJson(route, evaluation.regressions || [])
  );
  await page.route(/\/eval\/runs\/\d+\/rerun$/, async (route) => {
    const path = new URL(route.request().url()).pathname;
    const body = route.request().postDataJSON();
    state.requests.push({ path, method: route.request().method(), body });
    await fulfillJson(route, evaluation.createdRerun || {
      id: 91,
      dataset_id: 31,
      status: "pending",
      evaluation_scope: "subset",
      sample_filter: JSON.stringify(body.sample_ids),
      source_run_id: Number(path.split("/").at(-2)),
      comparison_role: "diagnostic",
      created_at: timestamp,
      completed_at: null,
    }, 202);
  });
  await page.route(/\/documents\/\d+\/chunks\/\d+$/, (route) =>
    fulfillJson(route, evaluation.sourceChunk || {
      document_id: 5,
      document_name: "public-source.md",
      chunk_index: 2,
      citation_id: "D5:C2",
      content: "公开数据集中的可核验原文。",
      page_start: 3,
      page_end: 3,
      paragraph_start: 4,
      paragraph_end: 4,
      char_start: 42,
      char_end: 55,
      locator_version: "extracted_text_v1",
      source_excerpt: "前置上下文。公开数据集中的可核验原文。后置上下文。",
      highlight_start: 6,
      highlight_end: 19,
      source_version: "2026-07",
      source_status: "current",
      jump_url: "/documents/5/chunks/2",
    })
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
