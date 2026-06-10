import { C, base, bulletList, callout, text } from "./common.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 1, "MOTIVATION", "为什么在 CSWin-UNet 上加频率注意力");
  bulletList(slide, ctx, 82, 160, 610, [
    "医学分割里，目标边界常常细、弱、形态不规则。",
    "浅层 skip 含有更多局部边界，但也更容易混入噪声和伪边界。",
    "CSWin-UNet 是强基底，但原结构没有显式频率增强或边界抑制机制。"
  ], { size: 22, gap: 76 });
  callout(slide, ctx, 760, 158, 330, 124, "设计问题",
    "能否只在 skip feature 上增强有用的高频边界，同时保留低频结构，不破坏 CSWin-UNet 主干？");
  callout(slide, ctx, 760, 318, 330, 124, "当前尝试",
    "借用 PFESA 的 FFT 高频/低频解耦，将其接入 x1/x2/x3 skip 做组合消融。");
  text(slide, ctx, 82, 548, 850, 44, "讲解重点：这是轻量插入模块，不是重新设计整套网络。", {
    size: 21, color: C.ink, bold: true,
  });
  return slide;
}
