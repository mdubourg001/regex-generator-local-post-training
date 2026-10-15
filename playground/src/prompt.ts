// Port de scripts/common.py : le prompt doit être identique à celui de l'entraînement, au caractère près.

const SYSTEM_PROMPT =
  "You write JavaScript regular expressions (ECMAScript 2018). The regex will validate strings with " +
  "RegExp.prototype.test: it must accept every valid string and reject every other one. Given a description, and " +
  "sometimes example strings, answer with a single regex literal such as /^\\d+$/i, with no explanation.";

const ALLOWED_FLAGS = new Set("gimsuy");
const LINE_TERMINATORS = /[\n\r\u2028\u2029]/;

export type Request = { description: string; match: string[]; noMatch: string[] };

function buildUserPrompt({ description, match, noMatch }: Request): string {
  // Chaînes en JSON, comme json.dumps(ensure_ascii=False) côté Python.
  const list = (items: string[]) => items.map((s) => `- ${JSON.stringify(s)}`).join("\n");
  const sections = [`Description: ${description}`];
  if (match.length) sections.push(`Should match:\n${list(match)}`);
  if (noMatch.length) sections.push(`Should not match:\n${list(noMatch)}`);
  return sections.join("\n\n");
}

/** Gabarit de chat Qwen3 avec enable_thinking=False (vérifié contre apply_chat_template de transformers). */
export function buildPrompt(request: Request): string {
  return (
    `<|im_start|>system\n${SYSTEM_PROMPT}<|im_end|>\n` +
    `<|im_start|>user\n${buildUserPrompt(request)}<|im_end|>\n` +
    `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  );
}

/** `/pattern/flags` → [pattern, flags]. null si le texte n'est pas exactement un littéral regex JS ES2018. */
function parseLiteral(text: string): [string, string] | null {
  if (text.length < 3 || text[0] !== "/" || LINE_TERMINATORS.test(text)) return null;
  let i = 1;
  let inClass = false;
  for (; i < text.length; i++) {
    const c = text[i];
    if (c === "\\") {
      i++;
      continue;
    }
    if (c === "[") inClass = true;
    else if (c === "]") inClass = false;
    else if (c === "/" && !inClass) break;
  }
  if (i >= text.length) return null;
  const pattern = text.slice(1, i);
  const flags = text.slice(i + 1);
  if (!pattern || [...flags].some((f) => !ALLOWED_FLAGS.has(f)) || new Set(flags).size !== flags.length) return null;
  return [pattern, flags];
}

/** Même tolérance que parse_answer : bloc de code, backticks ou guillemets autour du littéral, mais pas de texte. */
export function parseAnswer(raw: string): [string, string] | null {
  let text = raw.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  const fence = text.match(/^```[\w-]*\n([\s\S]*?)\n?```$/);
  if (fence) text = fence[1].trim();
  if (text.length >= 2 && text[0] === text.at(-1) && "`\"'".includes(text[0])) text = text.slice(1, -1).trim();
  return parseLiteral(text);
}
