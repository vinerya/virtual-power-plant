// Map a JSON-Pointer path (e.g. "/optimizer/gap_tolerance", "/sites/1/name")
// to the 1-based line of that key in a YAML document, without a full YAML
// AST. Handles the block-style YAML the config editor produces (nested maps
// and "- " sequences); falls back to the deepest ancestor found, or line 1.

function unescapePointer(seg: string): string {
  return seg.replace(/~1/g, "/").replace(/~0/g, "~");
}

interface Line {
  n: number; // 1-based
  indent: number;
  text: string; // trimmed-left content
}

function significantLines(yaml: string): Line[] {
  const out: Line[] = [];
  yaml.split(/\r?\n/).forEach((raw, i) => {
    const text = raw.replace(/^\s+/, "");
    if (!text || text.startsWith("#") || text === "---") return;
    out.push({ n: i + 1, indent: raw.length - text.length, text });
  });
  return out;
}

function keyOf(text: string): string | null {
  const m = /^(?:"([^"]+)"|'([^']+)'|([^:#\s][^:#]*?))\s*:(\s|$)/.exec(text);
  if (!m) return null;
  return m[1] ?? m[2] ?? m[3] ?? null;
}

export function pointerToLine(yaml: string, pointer: string): number {
  const lines = significantLines(yaml);
  if (!lines.length) return 1;
  const segs = pointer
    .replace(/^\$/, "")
    .split("/")
    .filter((s) => s !== "")
    .map(unescapePointer);

  // Search window: [start, end) over `lines`, children deeper than parentIndent.
  let start = 0;
  let end = lines.length;
  let parentIndent = -1;
  let found = lines[0].n;
  let inItem = false;

  for (const seg of segs) {
    let hit = -1;
    let hitIndent = -1;
    // Children of the current node share the smallest indent > parentIndent.
    let childIndent = Infinity;
    for (let i = start; i < end; i++) {
      if (lines[i].indent > parentIndent) childIndent = Math.min(childIndent, lines[i].indent);
    }
    const isIndex = /^\d+$/.test(seg);
    if (isIndex) {
      let count = -1;
      for (let i = start; i < end; i++) {
        const l = lines[i];
        if (l.indent === childIndent && l.text.startsWith("-")) {
          count += 1;
          if (count === Number(seg)) {
            hit = i;
            // Content of "- key: v" items sits 2 columns further in.
            hitIndent = l.indent;
            break;
          }
        }
      }
    } else {
      for (let i = start; i < end; i++) {
        const l = lines[i];
        const onDash = l.text.startsWith("- ");
        // Inside a sequence item the first key sits on the dash line
        // ("- name: x") and the rest two columns past the dash.
        const atLevel = inItem
          ? onDash
            ? l.indent === childIndent
            : l.indent === childIndent + 2
          : !onDash && l.indent === childIndent;
        if (!atLevel) continue;
        const text = onDash ? l.text.slice(2) : l.text;
        if (keyOf(text) === seg) {
          hit = i;
          hitIndent = onDash ? l.indent + 2 : l.indent;
          break;
        }
      }
    }
    if (hit === -1) break;
    found = lines[hit].n;
    if (isIndex) {
      // The item spans the dash line plus everything indented past the dash.
      let blockEnd = hit + 1;
      while (blockEnd < end && lines[blockEnd].indent > lines[hit].indent) blockEnd++;
      start = hit;
      end = blockEnd;
      parentIndent = lines[hit].indent - 1;
      inItem = true;
      continue;
    }
    let blockEnd = hit + 1;
    while (blockEnd < end && lines[blockEnd].indent > hitIndent) blockEnd++;
    start = hit + 1;
    end = blockEnd;
    parentIndent = hitIndent;
    inItem = false;
  }
  return found;
}
