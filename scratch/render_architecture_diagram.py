import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Artifact 디렉토리 경로
ARTIFACT_DIR = r"C:\Users\wolf\.gemini\antigravity-ide\brain\e205d749-95e5-4fca-873d-b84d4d5be0b8"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
IMG_PATH = os.path.join(ARTIFACT_DIR, "hybrid_timeframe_architecture.png")

# 다크 모드 프리미엄 스타일 설정
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#0f172a')

# 축 숨기기
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

# 타이틀
ax.text(7, 6.4, "RADE Multi-Timeframe Hybrid Architecture", fontsize=18, weight='bold', ha='center', color='#38bdf8')
ax.text(7, 6.0, "1H Macro Regime Filter + 15m Micro Execution Layer", fontsize=12, ha='center', color='#94a3b8')

# 박스 1: 1H HMM 국면 관리자 (Macro Layer)
box1 = patches.FancyBboxPatch((0.8, 1.8), 3.6, 3.6, boxstyle="round,pad=0.2", fc="#1e293b", ec="#38bdf8", lw=2)
ax.add_patch(box1)
ax.text(2.6, 5.0, "1H HMM Macro Layer", fontsize=14, weight='bold', ha='center', color='#38bdf8')
ax.text(2.6, 4.4, "• 3-State Gaussian HMM\n• Weekly Calendar Anchor\n• Regime: BULL / RANGE / BEAR\n• False Signal Filtering", fontsize=10, ha='center', color='#e2e8f0', linespacing=1.6)
ax.text(2.6, 2.3, "Role: 'The Forest' (Market Context)", fontsize=9, weight='bold', ha='center', color='#0284c7')

# 화살표 1 -> 2
arrow1 = patches.FancyArrowPatch((4.7, 3.6), (5.5, 3.6), arrowstyle='->,head_width=0.4,head_length=0.4', color='#38bdf8', lw=2.5)
ax.add_patch(arrow1)
ax.text(5.1, 3.9, "Regime\nState", fontsize=9, ha='center', color='#38bdf8', weight='bold')

# 박스 2: 15m 마이크로 실행 레이어 (Micro Execution)
box2 = patches.FancyBboxPatch((5.7, 1.8), 3.6, 3.6, boxstyle="round,pad=0.2", fc="#1e293b", ec="#10b981", lw=2)
ax.add_patch(box2)
ax.text(7.5, 5.0, "15m Micro Execution", fontsize=14, weight='bold', ha='center', color='#10b981')
ax.text(7.5, 4.4, "• Trend Pullback (15m 눌림목)\n• Micro Breakout Detection\n• Dynamic Position Sizing\n• 2~3x More Trade Opportunities", fontsize=10, ha='center', color='#e2e8f0', linespacing=1.6)
ax.text(7.5, 2.3, "Role: 'The Trees' (Precise Entry)", fontsize=9, weight='bold', ha='center', color='#059669')

# 화살표 2 -> 3
arrow2 = patches.FancyArrowPatch((9.6, 3.6), (10.4, 3.6), arrowstyle='->,head_width=0.4,head_length=0.4', color='#10b981', lw=2.5)
ax.add_patch(arrow2)
ax.text(10.0, 3.9, "Order\nFill", fontsize=9, ha='center', color='#10b981', weight='bold')

# 박스 3: 동적 청산 및 리스크 관리 (Exit & Risk Layer)
box3 = patches.FancyBboxPatch((10.6, 1.8), 2.8, 3.6, boxstyle="round,pad=0.2", fc="#1e293b", ec="#f59e0b", lw=2)
ax.add_patch(box3)
ax.text(12.0, 5.0, "Risk & Exit", fontsize=14, weight='bold', ha='center', color='#f59e0b')
ax.text(12.0, 4.4, "• Dynamic ATR 4.5x\n• 24h Time Stop\n• 80:20 TP / Breakeven\n• Strict Kill Switch", fontsize=10, ha='center', color='#e2e8f0', linespacing=1.6)
ax.text(12.0, 2.3, "Target: PF >= 1.80", fontsize=9, weight='bold', ha='center', color='#d97706')

# 하단 요약 배너
bottom_box = patches.FancyBboxPatch((0.8, 0.4), 12.6, 0.9, boxstyle="round,pad=0.1", fc="#1e1b4b", ec="#6366f1", lw=1.5)
ax.add_patch(bottom_box)
ax.text(7.1, 0.85, "Expected Outcome: Preserve High SNR (Win Rate 55%+, PF 1.80) while Expanding Trades (40/yr -> 80~120/yr)", fontsize=11, weight='bold', ha='center', color='#c7d2fe')

plt.tight_layout()
plt.savefig(IMG_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print("Saved diagram to:", IMG_PATH)
