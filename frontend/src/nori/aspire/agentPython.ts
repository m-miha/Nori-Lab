import type { TaskEvent } from "./api";

function literal(value: unknown): string {
  if (value === null) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (Array.isArray(value)) return "[" + value.map(literal).join(", ") + "]";
  if (typeof value === "object") {
    return "{" + Object.entries(value).map(([k, v]) => literal(k) + ": " + literal(v)).join(", ") + "}";
  }
  return JSON.stringify(value);
}

const keywords = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" "));
const identifier = (s: string) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(s) && !keywords.has(s) && s !== "__debug__";

function call(name: string, args: Record<string, unknown>): string {
  // Keep unexpected model-supplied names as data, including names that aren't Python identifiers.
  const fn = identifier(name) && !name.startsWith("_") ? "tools." + name : "tools[" + literal(name) + "]";
  const params = Object.keys(args).every(identifier)
    ? Object.entries(args).map(([k, v]) => k + "=" + literal(v)).join(", ")
    : "**" + literal(args);
  return "    " + fn + "(" + params + ")";
}

export function agentPython(events: TaskEvent[]): string {
  const lines: string[] = [];
  let pending: string | undefined;
  let instructionDone = false;
  for (const e of events) {
    switch (e.type) {
      case "step_start":
      case "step": {
        if (typeof e.tool !== "string") break;
        // The worker calls tools serially. Older traces can have just the completion event.
        if (e.type === "step_start" || pending !== e.tool) {
          lines.push(call(e.tool, (e.args ?? {}) as Record<string, unknown>));
        }
        pending = e.type === "step_start" ? e.tool : undefined;
        break;
      }
      case "say": if (typeof e.text === "string") lines.push(call("say", { text: e.text })); break;
      case "needs_input": if (typeof e.question === "string") lines.push(call("ask", { question: e.question })); break;
      case "skill_read": if (typeof e.name === "string") lines.push(call("read_skill", { name: e.name })); break;
      case "refused": lines.push("    # Refused: " + literal(e.reason)); break;
      case "continued": instructionDone = false; break;
      case "instruction_done":
      case "done": {
        // The worker only exposes these done() arguments through the instruction outcome.
        if (instructionDone || typeof e.success_claim !== "boolean") break;
        const args: Record<string, unknown> = { success: e.success_claim };
        if (e.summary) args.note = e.summary;
        if (typeof e.self_confidence === "number") args.confidence = e.self_confidence;
        lines.push(call("done", args));
        instructionDone = true;
        break;
      }
    }
  }
  return [
    '"""High-level tool requests recorded in the Nori Agent transcript."""',
    "",
    "",
    "def agent_run(dispatch):",
    '    """Pass a dispatcher that accepts a tool name and its argument dictionary."""',
    "    tools = _Tools(dispatch)",
    ...lines,
    "",
    "",
    "class _Tools:",
    "    def __init__(self, dispatch):",
    "        self._dispatch = dispatch",
    "",
    "    def __getitem__(self, name):",
    "        return lambda **arguments: self._dispatch(name, arguments)",
    "",
    "    __getattr__ = __getitem__",
    "",
  ].join("\n");
}
