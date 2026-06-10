import { C, base, callout, text } from "./common.mjs";

export async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 9, "INTERPRETATION", "结果说明：多尺度 skip 增强更稳，单点增强不一定有效");
  callout(slide, ctx, 86, 150, 330, 150, "综合最优：x123",
    "Dice 相比 baseline +0.001014，IoU 和 HD95 也有小幅收益。说明三条 skip 同时加入 PFESA 后，整体分割更稳定。",
    C.green, C.greenSoft);
  callout(slide, ctx, 462, 150, 330, 150, "边界最优：x13",
    "Boundary IoU = 0.340406。浅层细节与深层语义组合，可能更利于边界区域。",
    C.orange, C.orangeSoft);
  callout(slide, ctx, 838, 150, 330, 150, "负例提醒：x1/x3",
    "单独启用 x1 或 x3 低于 baseline，说明频率增强需要控制位置，浅层高频可能放大噪声。",
    C.purple, C.purpleSoft);
  text(slide, ctx, 108, 390, 930, 44, "当前比较更适合作为“模块位置选择”的探索实验，而不是最终主实验结论。", {
    size: 25, bold: true, color: C.ink, align: "center",
  });
  text(slide, ctx, 156, 490, 800, 36, "讲解时可以用这个逻辑：先承认提升幅度小，再强调它告诉我们 PFESA 放置位置的趋势。", {
    size: 20, color: C.muted, align: "center",
  });
  return slide;
}
