import { CopilotRuntime, OpenAIAdapter } from "@copilotkit/runtime";

export async function POST(req: Request) {
  const copilotRuntime = new CopilotRuntime();
  // Simple pass-through to OpenAI or other adapter
  return copilotRuntime.response(req, new OpenAIAdapter());
}
