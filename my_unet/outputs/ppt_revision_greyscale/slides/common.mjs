export const C = {
  bg: "#F7F8FA",
  paper: "#FFFFFF",
  ink: "#171717",
  dark: "#2B2B2B",
  mid: "#666666",
  soft: "#E6E8EB",
  softer: "#F0F1F3",
  line: "#C9CDD3",
  lightLine: "#E1E4E8",
};

export const FONT = {
  title: "Microsoft YaHei UI",
  body: "Microsoft YaHei",
  mono: "Consolas",
};

export const ARCH_IMG = "F:/code/git/segmentation/my_uet/汇报/模型架构图.png";

export function base(slide, ctx, n, kicker, title) {
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 8, fill: C.dark, line: ctx.line() });
  ctx.addText(slide, {
    x: 60, y: 34, w: 230, h: 24, text: kicker, fontSize: 12, color: C.mid,
    bold: true, typeface: FONT.body, valign: "mid",
  });
  ctx.addText(slide, {
    x: 60, y: 62, w: 960, h: 56, text: title, fontSize: 33, color: C.ink,
    bold: true, typeface: FONT.title, valign: "mid",
  });
  ctx.addShape(slide, { x: 60, y: 646, w: 1060, h: 1, fill: C.line, line: ctx.line() });
  ctx.addText(slide, {
    x: 1110, y: 664, w: 90, h: 24, text: `${String(n).padStart(2, "0")}/05`,
    fontSize: 12, color: C.mid, typeface: FONT.body, align: "right",
  });
}

export function text(slide, ctx, x, y, w, h, value, opts = {}) {
  const shape = ctx.addText(slide, {
    x, y, w, h, text: value, fontSize: opts.size ?? 20, color: opts.color ?? C.ink,
    bold: opts.bold ?? false, typeface: opts.face ?? FONT.body, valign: opts.valign ?? "top",
    align: opts.align ?? "left", fill: opts.fill ?? "#00000000",
    line: opts.line ?? ctx.line(),
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  });
  return shape;
}

export function box(slide, ctx, x, y, w, h, opts = {}) {
  return ctx.addShape(slide, {
    x, y, w, h, geometry: "rect", fill: opts.fill ?? C.paper,
    line: opts.line ?? { style: "solid", fill: opts.stroke ?? C.line, width: opts.width ?? 1 },
  });
}

export function labelBox(slide, ctx, x, y, w, h, label, opts = {}) {
  box(slide, ctx, x, y, w, h, { fill: opts.fill ?? C.paper, stroke: opts.stroke ?? C.line, width: opts.width ?? 1 });
  text(slide, ctx, x + 12, y + 8, w - 24, h - 16, label, {
    size: opts.size ?? 18, color: opts.color ?? C.ink, bold: opts.bold ?? false,
    align: opts.align ?? "center", valign: "mid",
  });
}

export function bulletList(slide, ctx, x, y, w, items, opts = {}) {
  const size = opts.size ?? 21;
  const gap = opts.gap ?? 58;
  items.forEach((item, i) => {
    ctx.addShape(slide, { x, y: y + i * gap + 11, w: 8, h: 8, geometry: "ellipse", fill: C.dark, line: ctx.line() });
    text(slide, ctx, x + 22, y + i * gap, w - 22, gap - 4, item, { size, color: opts.color ?? C.ink });
  });
}

export function callout(slide, ctx, x, y, w, h, title, body, opts = {}) {
  box(slide, ctx, x, y, w, h, { fill: opts.fill ?? C.paper, stroke: opts.stroke ?? C.line, width: opts.width ?? 1 });
  text(slide, ctx, x + 18, y + 14, w - 36, 26, title, { size: opts.titleSize ?? 18, bold: true, color: C.ink });
  text(slide, ctx, x + 18, y + 46, w - 36, h - 58, body, { size: opts.bodySize ?? 16, color: C.dark });
}

export function formulaBox(slide, ctx, x, y, w, h, title, formula, note) {
  box(slide, ctx, x, y, w, h, { fill: C.paper, stroke: C.dark, width: 1.2 });
  text(slide, ctx, x + 14, y + 10, w - 28, 22, title, { size: 14, bold: true, color: C.ink });
  text(slide, ctx, x + 14, y + 38, w - 28, h - 72, formula, { size: 12.5, color: C.ink, face: FONT.mono });
  text(slide, ctx, x + 14, y + h - 30, w - 28, 20, note, { size: 11, color: C.mid });
}
