import { C, ARCH_IMG, base, text, callout } from "./common.mjs";

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 4, "ARCHITECTURE", "整体模型架构图：PFESA 作用在 skip feature");
  await ctx.addImage(slide, {
    path: ARCH_IMG, x: 42, y: 126, w: 880, h: 495, fit: "contain",
    alt: "CSWin-UNet with PFESA architecture figure",
  });
  callout(slide, ctx, 946, 150, 250, 120, "图的读法 1",
    "三条 skip 都画 PFESA 时，对应 x123 结构示意。",
    C.purple, C.purpleSoft);
  callout(slide, ctx, 946, 296, 250, 120, "图的读法 2",
    "实际消融中，x12/x13/x23 只启用对应 skip，其余 skip 保持原始连接。",
    C.blue, C.blueSoft);
  callout(slide, ctx, 946, 442, 250, 120, "图的读法 3",
    "decoder 是连续上采样路径，不是三条并行 decoder。",
    C.orange, C.orangeSoft);
  text(slide, ctx, 64, 628, 910, 24, "注意：如果图中个别通道标注和代码不一致，以代码流程 C4 -> C3 -> C2 -> C1 为准。", {
    size: 16, color: C.muted,
  });
  return slide;
}
