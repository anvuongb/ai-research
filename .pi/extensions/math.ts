/**
 * math.ts — render LaTeX math as readable Unicode in the Pi terminal transcript.
 *
 * Pi exposes `registerMarkdownTransformer`, which rewrites user/assistant Markdown
 * before it is rendered in the interactive transcript. We use it to convert:
 *
 *   - inline `$...$` and `\(...\)`
 *   - display `$$...$$` and `\[...\]`
 *   - fenced ```math / ```latex / ```tex blocks
 *
 * into Unicode text (Greek letters, operators, sub/superscripts, …), so equations
 * are legible in terminals without image support. Code spans and non-math fenced
 * blocks are left untouched.
 *
 * This is intentionally lossy (it is a terminal preview, not a typesetter): heavy
 * matrices, alignment, and unknown macros degrade to readable plain text.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// ---------------------------------------------------------------------------
// Symbol tables
// ---------------------------------------------------------------------------

const GREEK: Record<string, string> = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε",
  zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ",
  lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", omicron: "ο", pi: "π", varpi: "ϖ",
  rho: "ρ", varrho: "ϱ", sigma: "σ", varsigma: "ς", tau: "τ", upsilon: "υ",
  phi: "φ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω",
  Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π",
  Sigma: "Σ", Upsilon: "Υ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
};

const SYMBOLS: Record<string, string> = {
  sum: "Σ", prod: "Π", coprod: "∐", int: "∫", oint: "∮", iint: "∬",
  infty: "∞", partial: "∂", nabla: "∇", ell: "ℓ", hbar: "ℏ", Re: "ℜ", Im: "ℑ",
  approx: "≈", neq: "≠", ne: "≠", geq: "≥", ge: "≥", leq: "≤", le: "≤",
  equiv: "≡", sim: "∼", simeq: "≃", cong: "≅", propto: "∝", ll: "≪", gg: "≫",
  in: "∈", notin: "∉", ni: "∋", subset: "⊂", subseteq: "⊆", supset: "⊃",
  supseteq: "⊇", cup: "∪", cap: "∩", setminus: "∖", emptyset: "∅", varnothing: "∅",
  forall: "∀", exists: "∃", nexists: "∄", neg: "¬", lnot: "¬",
  wedge: "∧", land: "∧", vee: "∨", lor: "∨",
  to: "→", rightarrow: "→", leftarrow: "←", leftrightarrow: "↔",
  Rightarrow: "⇒", Leftarrow: "⇐", Leftrightarrow: "⇔", implies: "⟹",
  mapsto: "↦", uparrow: "↑", downarrow: "↓", hookrightarrow: "↪",
  cdot: "·", cdots: "⋯", dots: "…", ldots: "…", vdots: "⋮", ddots: "⋱",
  times: "×", div: "÷", pm: "±", mp: "∓", circ: "∘", bullet: "•",
  ast: "∗", star: "⋆", oplus: "⊕", otimes: "⊗", odot: "⊙", ominus: "⊖",
  langle: "⟨", rangle: "⟩", lceil: "⌈", rceil: "⌉", lfloor: "⌊", rfloor: "⌋",
  mid: "|", vert: "|", Vert: "‖", lVert: "‖", rVert: "‖",
  angle: "∠", perp: "⊥", parallel: "∥", prime: "′", degree: "°",
  because: "∵", therefore: "∴", square: "□", checkmark: "✓",
  log: "log", ln: "ln", exp: "exp", sin: "sin", cos: "cos", tan: "tan",
  max: "max", min: "min", argmax: "argmax", argmin: "argmin", det: "det",
  lim: "lim", sup: "sup", inf: "inf", gcd: "gcd", Pr: "Pr", bmod: "mod", mod: "mod",
  // layout / font commands are simply dropped
  mathrm: "", mathbf: "", mathcal: "", mathbb: "", boldsymbol: "", bm: "",
  operatorname: "", text: "", textrm: "", textbf: "", textit: "", mbox: "",
  left: "", right: "", big: "", Big: "", bigg: "", Bigg: "", bigl: "", bigr: "",
  Bigl: "", Bigr: "", biggl: "", biggr: "", displaystyle: "", textstyle: "",
  quad: "  ", qquad: "    ", limits: "", nolimits: "",
};

const DOUBLE_STRUCK: Record<string, string> = {
  E: "𝔼", R: "ℝ", N: "ℕ", Z: "ℤ", Q: "ℚ", C: "ℂ", P: "ℙ", D: "𝔻", S: "𝕊",
  "1": "𝟙", "0": "𝟘",
};

const SCRIPT: Record<string, string> = {
  A: "𝒜", B: "ℬ", D: "𝒟", E: "ℰ", F: "ℱ", H: "ℋ", I: "ℐ", L: "ℒ", M: "ℳ",
  N: "𝒩", O: "𝒪", P: "𝒫", R: "ℛ", S: "𝒮", T: "𝒯", U: "𝒰", V: "𝒱", W: "𝒲",
  X: "𝒳",
};

const ACCENTS: Record<string, string> = {
  hat: "\u0302", widehat: "\u0302", bar: "\u0304", overline: "\u0304",
  tilde: "\u0303", widetilde: "\u0303", vec: "\u20D7", dot: "\u0307", ddot: "\u0308",
};

const SUBSCRIPT: Record<string, string> = {
  "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
  "7": "₇", "8": "₈", "9": "₉",
  "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
  a: "ₐ", e: "ₑ", h: "ₕ", i: "ᵢ", j: "ⱼ", k: "ₖ", l: "ₗ", m: "ₘ", n: "ₙ",
  o: "ₒ", p: "ₚ", r: "ᵣ", s: "ₛ", t: "ₜ", u: "ᵤ", v: "ᵥ", x: "ₓ",
};

const SUPERSCRIPT: Record<string, string> = {
  "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
  "7": "⁷", "8": "⁸", "9": "⁹",
  "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
  a: "ᵃ", b: "ᵇ", c: "ᶜ", d: "ᵈ", e: "ᵉ", f: "ᶠ", g: "ᵍ", h: "ʰ", i: "ⁱ",
  j: "ʲ", k: "ᵏ", l: "ˡ", m: "ᵐ", n: "ⁿ", o: "ᵒ", p: "ᵖ", r: "ʳ", s: "ˢ",
  t: "ᵗ", u: "ᵘ", v: "ᵛ", w: "ʷ", x: "ˣ", y: "ʸ", z: "ᶻ",
  A: "ᴬ", B: "ᴮ", D: "ᴰ", E: "ᴱ", G: "ᴳ", H: "ᴴ", I: "ᴵ", J: "ᴶ", K: "ᴷ",
  L: "ᴸ", M: "ᴹ", N: "ᴺ", O: "ᴼ", P: "ᴾ", R: "ᴿ", T: "ᵀ", U: "ᵁ", V: "ⱽ", W: "ᵂ",
};

// A brace group that may contain one level of nesting, e.g. `\partial \mathcal{L}`.
const GROUP = String.raw`\{((?:[^{}]|\{[^{}]*\})*)\}`;

function toScript(body: string, map: Record<string, string>): string | null {
  let out = "";
  for (const ch of body) {
    const mapped = map[ch];
    if (mapped === undefined) return null;
    out += mapped;
  }
  return out;
}

function boldify(s: string): string {
  return Array.from(s)
    .map((ch) => {
      const c = ch.codePointAt(0) as number;
      if (c >= 65 && c <= 90) return String.fromCodePoint(0x1d400 + (c - 65));
      if (c >= 97 && c <= 122) return String.fromCodePoint(0x1d41a + (c - 97));
      if (c >= 48 && c <= 57) return String.fromCodePoint(0x1d7ce + (c - 48));
      return ch;
    })
    .join("");
}

// ---------------------------------------------------------------------------
// LaTeX -> Unicode
// ---------------------------------------------------------------------------

export function latexToUnicode(latex: string): string {
  let s = latex.trim();

  // LaTeX line break
  s = s.replace(/\\\\/g, "\n");

  // spacing commands
  s = s.replace(/\\[,;:]\s?/g, " ");
  s = s.replace(/\\!/g, "");
  s = s.replace(/\\ /g, " ");

  // escaped symbols (incl. \| = double bar)
  s = s.replace(/\\([%&_#{}$])/g, "$1");
  s = s.replace(/\\\|/g, "‖");

  // fractions (brace-nested-aware, then bare single chars)
  s = s.replace(new RegExp(String.raw`\\[dt]?frac\s*` + GROUP + String.raw`\s*` + GROUP, "g"), "($1)/($2)");
  s = s.replace(/\\[dt]?frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])/g, "$1/$2");

  // roots
  s = s.replace(new RegExp(String.raw`\\sqrt\s*\[([^{}]*)\]\s*` + GROUP, "g"), "($1)√($2)");
  s = s.replace(new RegExp(String.raw`\\sqrt\s*` + GROUP, "g"), "√($1)");

  // accents / decorators
  s = s.replace(
    new RegExp(String.raw`\\(hat|widehat|bar|overline|tilde|widetilde|vec|dot|ddot)\s*` + GROUP, "g"),
    (_m, acc: string, body: string) => body + (ACCENTS[acc] ?? ""),
  );

  // font wrappers
  s = s.replace(new RegExp(String.raw`\\mathbb\s*` + GROUP, "g"), (_m, b: string) =>
    Array.from(b).map((c) => DOUBLE_STRUCK[c] ?? c).join(""));
  s = s.replace(new RegExp(String.raw`\\mathcal\s*` + GROUP, "g"), (_m, b: string) =>
    Array.from(b).map((c) => SCRIPT[c] ?? c).join(""));
  s = s.replace(new RegExp(String.raw`\\mathbf\s*` + GROUP, "g"), (_m, b: string) => boldify(b));
  s = s.replace(
    new RegExp(String.raw`\\(?:boldsymbol|bm|mathrm|mathsf|mathtt|text|textrm|textbf|textit|operatorname|mbox)\s*` + GROUP, "g"),
    "$1",
  );

  // sub/superscripts (brace-nested-aware, then bare single chars)
  s = s.replace(new RegExp(String.raw`([_^])\s*` + GROUP, "g"), (m, op: string, body: string) => {
    const conv = toScript(body, op === "_" ? SUBSCRIPT : SUPERSCRIPT);
    if (conv !== null) return conv;
    return /^[A-Za-z0-9]+$/.test(body) ? `${op}${body}` : `${op}(${body})`;
  });
  s = s.replace(/([_^])\s*([A-Za-z0-9])/g, (m, op: string, ch: string) => {
    const map = op === "_" ? SUBSCRIPT : SUPERSCRIPT;
    const conv = map[ch];
    return conv !== undefined ? conv : m;
  });

  // remaining named commands
  s = s.replace(/\\([A-Za-z]+)/g, (m, name: string) => {
    if (name in GREEK) return GREEK[name];
    if (name in SYMBOLS) return SYMBOLS[name];
    return name; // unknown macro: drop the backslash, keep the name
  });

  // leftover grouping braces
  s = s.replace(/[{}]/g, "");

  // tidy: collapse runs of spaces but keep newlines
  s = s.split("\n").map((line) => line.replace(/[ \t]+/g, " ").trim()).join("\n").trim();
  return s;
}

// ---------------------------------------------------------------------------
// Markdown transformer
// ---------------------------------------------------------------------------

// Order matters: fenced code, inline code, display math, inline math. The inline
// `$...$` alternative requires the body to look math-ish, so stray prose dollars
// (e.g. "$PATH") are not swallowed.
const MASTER =
  /(`{3,})([^\n]*)\n([\s\S]*?)(?:\n\1|$)|(`[^`\n]*`)|(\$\$([\s\S]+?)\$\$)|(\\\[([\s\S]+?)\\\])|(\\\(([\s\S]+?)\\\))|(\$([^\$\n]*[\\_^{}=+\-<>|][^\$\n]*|[A-Za-z])\$)/g;

function looksLikeMath(body: string): boolean {
  if (body.length === 0) return false;
  if (/^\s/.test(body) || /\s$/.test(body)) return false; // "$ 5 $" style
  if (/^\d+([.,]\d+)?$/.test(body)) return false; // plain currency amount
  return true;
}

export function transformMarkdown(markdown: string): string {
  return markdown.replace(
    MASTER,
    (
      match,
      fence: string | undefined,
      info: string | undefined,
      body: string | undefined,
      inlineCode: string | undefined,
      _dd: string | undefined,
      ddBody: string | undefined,
      _br: string | undefined,
      brBody: string | undefined,
      _par: string | undefined,
      parBody: string | undefined,
      _dol: string | undefined,
      dolBody: string | undefined,
    ) => {
      if (inlineCode !== undefined) return match; // leave code spans alone
      if (fence !== undefined) {
        const lang = (info ?? "").trim().toLowerCase();
        if (lang === "math" || lang === "latex" || lang === "tex") {
          return "```\n" + latexToUnicode(body ?? "") + "\n```";
        }
        return match; // leave all other code blocks alone
      }
      if (ddBody !== undefined) return "\n" + latexToUnicode(ddBody) + "\n";
      if (brBody !== undefined) return latexToUnicode(brBody);
      if (parBody !== undefined) return latexToUnicode(parBody);
      if (dolBody !== undefined) {
        return looksLikeMath(dolBody) ? latexToUnicode(dolBody) : match;
      }
      return match;
    },
  );
}

export default function (pi: ExtensionAPI): void {
  pi.registerMarkdownTransformer((markdown) => transformMarkdown(markdown));
}
