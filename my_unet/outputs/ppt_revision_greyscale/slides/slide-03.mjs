import { C, ARCH_IMG, base, text, formulaBox } from "./common.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 3, "ARCHITECTURE", "整体模型架构图：PFESA 作用在 skip feature");
  await ctx.addImage(slide, {
    path: ARCH_IMG, x: 42, y: 126, w: 802, h: 500, fit: "contain",
    alt: "CSWin-UNet with PFESA architecture figure",
  });
  formulaBox(slide, ctx, 870, 134, 330, 126, "CSWin stripe self-attention",
    "Q_n^i = W_n^Q X_h^i\nK_n^i = W_n^K X_h^i\nV_n^i = W_n^V X_h^i\nY_n^i = Softmax(Q_n^i(K_n^i)^T / sqrt(d_n)) V_n^i",
    "对应基底 CSWin Transformer Block");
  formulaBox(slide, ctx, 870, 284, 330, 118, "PFESA frequency decoupling",
    "G_σ(u,v) = exp(-(u^2+v^2)/(2σ^2))\nH_l = G_σ ⊙ F(X)\nH_h = (1 - G_σ) ⊙ F(X)",
    "对应 FFT 高频/低频解耦");
  formulaBox(slide, ctx, 870, 426, 330, 132, "PFESA attention",
    "EA = (X_h - μ_h)^2 / σ_h^2\nSA = Sigmoid((X_l^2 - μ_l) / σ_l^2)\nA = Sigmoid(EA + SA)\nX_out = A ⊙ X",
    "对应 skip feature 的频率增强");
  text(slide, ctx, 70, 632, 780, 22, "注意：PFESA 是后加在 skip feature 上的模块，不是 CSWin-UNet 原论文自带结构。", {
    size: 15, color: C.mid,
  });
  return slide;
}
