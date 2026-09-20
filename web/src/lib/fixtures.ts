/** Synthetic fixture clauses for local demos (from smoke_set). */

export type FixtureClause = {
  id: string;
  label: string;
  clause_text: string;
};

export const FIXTURES: FixtureClause[] = [
  {
    id: "indemnity-capped",
    label: "Indemnity (capped)",
    clause_text:
      "Supplier shall indemnify Customer against third-party claims arising from Supplier's breach of this Agreement, provided that Supplier's total liability under this section will not exceed the fees paid under this Agreement during the twelve months preceding the claim.",
  },
  {
    id: "indemnity-broad",
    label: "Indemnity (uncapped)",
    clause_text:
      "Vendor shall defend, indemnify, and hold harmless Company, its affiliates, officers, directors, employees, and customers from any and all claims, losses, liabilities, damages, costs, and expenses of every kind, whether direct or indirect, arising out of or related in any way to this Agreement, without limitation.",
  },
  {
    id: "liability-bilateral",
    label: "Liability cap",
    clause_text:
      "Except for either party's fraud, willful misconduct, or breach of confidentiality, each party's aggregate liability under this Agreement shall not exceed the fees paid or payable in the twelve months before the event giving rise to the claim, and neither party shall be liable for consequential or special damages.",
  },
  {
    id: "noncompete-broad",
    label: "Broad non-compete",
    clause_text:
      "For two years after the end of employment, Employee shall not, anywhere in the United States, directly or indirectly engage in, assist, invest in, or be employed by any business that competes in any respect with Company.",
  },
];
