# NAACL 2027 submission gate

Verified against the official NAACL 2027 main-conference call and the current
ACL Rolling Review (ARR) call on **2026-09-26**.

Official pages:

- NAACL 2027 main conference: https://2027.naacl.org/calls/main_conference_papers/
- ARR call for papers: https://aclrollingreview.org/cfp
- ARR author checklist: https://aclrollingreview.org/authorchecklist
- ARR submission fields: https://aclrollingreview.org/submissionform

## Frozen target dates

NAACL 2027 uses the ARR **October 2026** cycle.

- ARR submission deadline: **2026-10-12, 23:59 AoE**
- Reviewer-registration deadline for all authors: **2026-10-12**
- ARR meta-reviews released: **2026-12-18**
- NAACL commitment deadline: **2026-12-23**
- NAACL notification: **2027-02-10**
- Camera-ready deadline: **2027-03-03**
- Conference: **2027-06-01 to 2027-06-05**, San Francisco

## Paper-format gate

For a long paper:

- maximum **8 pages of main content**;
- Limitations is mandatory and placed after the conclusion, before references;
- optional Ethical Considerations may follow the conclusion/Limitations area;
- references are unlimited;
- appendices are allowed after references in the same PDF and normally remain
  double-column;
- use the official ACL style in review mode;
- run the ACL pubcheck before submission.

Reviewers are not required to read appendices. Any method detail needed to judge
technical correctness, novelty, claims, or implications must remain in the
8-page main paper.

## Anonymity gate

The ARR review remains two-way anonymized.

The submitted PDF and supplementary archives must not contain:

- author names or affiliations;
- acknowledgements;
- identity-revealing self-reference;
- a public non-anonymous repository link;
- local machine/user paths;
- tracking links such as Dropbox/Google Drive.

The public TrajUQ repository is therefore a development repository only. Build a
separate anonymized software ZIP for ARR rather than linking this repository in
the review PDF.

## October 2026 sustainable-reviewing requirement

ARR's sustainable-reviewing policy starts with the October 2026 cycle.
All authors need complete OpenReview profiles, and review is guaranteed only
when the submission contributes a qualified service contributor under the
current ARR policy. Complete the service/profile requirements early rather than
during the final upload day.

At minimum, every author should verify:

- complete OpenReview profile;
- current affiliation history;
- verified email(s);
- ORCID;
- conflicts;
- DBLP / ACL Anthology links where applicable;
- reviewer/service-contributor eligibility and registration.

## TrajUQ scientific gate before paper numbers

Do not populate the manuscript's result cells until:

1. LLaDA exact 20-prompt parity passes.
2. The 50-example GSM8K pilot passes the automated quality gate.
3. At least 10 pilot traces are manually reviewed.
4. The scorer audit is complete.
5. The full frozen model/dataset matrix is generated without changing the
   primary feature dictionary or test-label-dependent decisions.
6. Held-out calibration, nested comparison, feature ablations, repeated-sampling
   baseline, and macro cell aggregation complete.
7. Every reported number is traceable to immutable model/data/code revisions and
   artifact hashes.

## Submission-week checklist

Run:

```bash
python scripts/audit_paper.py --submission-ready
```

Then separately run the official ACL pubcheck on the compiled PDF.

Build the anonymous software archive with `python scripts/build_anonymous_supplement.py` and inspect its MANIFEST.json before upload.

Before upload verify:

- no TBD placeholders;
- no public GitHub URL in PDF or anonymous supplement;
- no acknowledgements;
- Limitations present and contains no new experiments/results;
- appendix follows references;
- all figures/tables remain readable at ACL size;
- all citations resolve;
- software ZIP is anonymized;
- data ZIP, if submitted, contains no gated/raw data that cannot be redistributed;
- Responsible NLP Research questions are answered accurately;
- all author OpenReview/service requirements are complete.
