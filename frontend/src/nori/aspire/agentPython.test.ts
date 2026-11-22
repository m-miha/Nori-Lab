import { expect, it } from "vitest";
import { agentPython } from "./agentPython";

it("writes each reported call once and preserves repeated calls, arguments and refusals", () => {
  const args = { side: "left", allow_contact: false, xyz: [1, null, 3] };
  const source = agentPython([
    { n: 0, type: "skill_read", name: "reach" },
    { n: 1, type: "step_start", tool: "move_to", args },
    { n: 2, type: "step", tool: "move_to", args },
    { n: 3, type: "refused", reason: "outside workspace" },
    { n: 4, type: "step", tool: "move_to", args },
    { n: 5, type: "needs_input", question: "Which cup?" },
    { n: 6, type: "say", text: 'The "red" one.\nOK.' },
    { n: 7, type: "instruction_done", success_claim: true, summary: "Reached" },
    { n: 8, type: "done", success_claim: true, summary: "Reached" },
  ]);
  const move = '    tools.move_to(side="left", allow_contact=False, xyz=[1, None, 3])';
  expect(source.split("\n").filter((line) => line.startsWith("    tools.") || line.startsWith("    #"))).toEqual([
    '    tools.read_skill(name="reach")', move, '    # Refused: "outside workspace"', move,
    '    tools.ask(question="Which cup?")', '    tools.say(text="The \\"red\\" one.\\nOK.")',
    '    tools.done(success=True, note="Reached")',
  ]);
});
