import { C, FONT, base, text, callout } from "./common.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 720, fill: C.paper, line: ctx.line() });
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 12, fill: C.blue, line: ctx.line() });
  text(slide, ctx, 70, 78, 980, 132, "CSWin-UNet with PFESA\non Configurable Skip Features", {
    size: 48, bold: true, face: FONT.title, color: C.ink,
  });
  text(slide, ctx, 74, 238, 880, 42, "频率增强 skip feature 的医学图像分割实验", {
    size: 26, color: C.purple, bold: true,
  });
  callout(slide, ctx, 74, 348, 760, 118, "一句话主旨",
    "在不改变 CSWin-UNet 主干的前提下，将 PFESA 作为无参数频率注意力模块插入 skip feature，增强 decoder 接收的边界和结构信息。",
    C.blue, C.blueSoft);
  callout(slide, ctx, 886, 348, 260, 118, "汇报重点",
    "结构改动\n消融设置\n验证集结果\n后续计划",
    C.purple, C.purpleSoft);
  text(slide, ctx, 74, 620, 900, 28, "CSWin-UNet + PFESA | 组会讲解辅助 PPT", { size: 16, color: C.muted });
  return slide;
}
