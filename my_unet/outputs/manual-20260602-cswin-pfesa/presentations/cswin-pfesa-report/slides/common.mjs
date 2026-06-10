export const C = {
  ink: "#172033",
  muted: "#657085",
  faint: "#EEF2F7",
  paper: "#FBFCFE",
  blue: "#2563EB",
  blueSoft: "#DBEAFE",
  purple: "#7C3AED",
  purpleSoft: "#EDE9FE",
  green: "#16A34A",
  greenSoft: "#DCFCE7",
  orange: "#F97316",
  orangeSoft: "#FFEDD5",
  red: "#DC2626",
  line: "#CBD5E1",
  white: "#FFFFFF",
};

export const FONT = {
  title: "Microsoft YaHei UI",
  body: "Microsoft YaHei",
  mono: "Consolas",
};

export const ARCH_IMG = "F:/code/git/segmentation/my_uet/汇报/模型架构图.png";

export function base(slide, ctx, n, kicker, title) {
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 720, fill: C.paper, line: ctx.line() });
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 10, fill: C.blue, line: ctx.line() });
  ctx.addText(slide, {
    x: 58, y: 34, w: 190, h: 24, text: kicker, fontSize: 13, color: C.blue,
    bold: true, typeface: FONT.body, valign: "mid",
  });
  ctx.addText(slide, {
    x: 58, y: 62, w: 920, h: 58, text: title, fontSize: 34, color: C.ink,
    bold: true, typeface: FONT.title, valign: "mid",
  });
  ctx.addText(slide, {
    x: 1138, y: 674, w: 70, h: 22, text: String(n).padStart(2, "0"), fontSize: 12,
    color: C.muted, typeface: FONT.body, align: "right",
  });
  ctx.addShape(slide, { x: 58, y: 650, w: 1064, h: 1, fill: C.line, line: ctx.line() });
}

export function text(slide, ctx, x, y, w, h, value, opts = {}) {
  return ctx.addText(slide, {
    x, y, w, h, text: value, fontSize: opts.size ?? 20, color: opts.color ?? C.ink,
    bold: opts.bold ?? false, typeface: opts.face ?? FONT.body, valign: opts.valign ?? "top",
    align: opts.align ?? "left", insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
    fill: opts.fill ?? "#00000000", line: opts.line ?? ctx.line(),
  });
}

export function box(slide, ctx, x, y, w, h, opts = {}) {
  return ctx.addShape(slide, {
    x, y, w, h, geometry: "rect", fill: opts.fill ?? C.white,
    line: opts.line ?? { style: "solid", fill: opts.stroke ?? C.line, width: opts.width ?? 1 },
  });
}

export function labelBox(slide, ctx, x, y, w, h, label, opts = {}) {
  box(slide, ctx, x, y, w, h, { fill: opts.fill ?? C.white, stroke: opts.stroke ?? C.line, width: opts.width ?? 1 });
  text(slide, ctx, x + 14, y + 10, w - 28, h - 18, label, {
    size: opts.size ?? 18, color: opts.color ?? C.ink, bold: opts.bold ?? false,
    align: opts.align ?? "center", valign: opts.valign ?? "mid",
  });
}

export function bulletList(slide, ctx, x, y, w, items, opts = {}) {
  const size = opts.size ?? 21;
  const gap = opts.gap ?? 42;
  items.forEach((item, i) => {
    ctx.addShape(slide, { x, y: y + i * gap + 8, w: 9, h: 9, geometry: "ellipse", fill: opts.dot ?? C.blue, line: ctx.line() });
    text(slide, ctx, x + 22, y + i * gap, w - 22, Math.max(28, gap - 8), item, { size, color: opts.color ?? C.ink });
  });
}

export function callout(slide, ctx, x, y, w, h, title, body, color = C.blue, fill = C.blueSoft) {
  box(slide, ctx, x, y, w, h, { fill, stroke: color, width: 1.2 });
  text(slide, ctx, x + 18, y + 15, w - 36, 28, title, { size: 19, bold: true, color });
  text(slide, ctx, x + 18, y + 50, w - 36, h - 62, body, { size: 17, color: C.ink });
}
