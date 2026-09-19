/** Process-local aggregate diagnostics. No predictions or truth are retained;
 * these counters do not affect learning, random streams or stopping. */
export class PredictionQuality {
  private total = 0;
  private nonconverged = 0;
  private abstained = 0;
  private reasons: Record<string, number> = {};
  record(values: Readonly<Record<string, number | null | "ambiguous">>, converged: boolean, reason: string): void {
    this.total++;
    if (!converged) this.nonconverged++;
    if (Object.values(values).some(v => typeof v !== "number")) this.abstained++;
    this.reasons[reason] = (this.reasons[reason] ?? 0) + 1;
  }
  snapshot() {
    return { predictions: this.total, nonconverged: this.nonconverged, anyOutputAbstained: this.abstained,
      nonconvergedRate: this.total ? this.nonconverged / this.total : null,
      anyOutputAbstentionRate: this.total ? this.abstained / this.total : null,
      terminationReasons: { ...this.reasons }, confidencePolicy: "converged-required" };
  }
  reset(): void { this.total = this.nonconverged = this.abstained = 0; this.reasons = {}; }
}
export const predictionQuality = new PredictionQuality();
