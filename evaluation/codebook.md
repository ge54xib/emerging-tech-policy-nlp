# Annotation Codebook — Relation Types
## NLI Evaluation (`annotation.json`)

Based on Ranga & Etzkowitz (2013). Each entry in `annotation.json` shows a sentence (`sent_text`) and two co-occurring actors (`entity_1`, `entity_2`). Assign the `true_relation` that best describes the relationship between the two actors **as expressed in that sentence**.

---

## Labels

### `technology_transfer`
**Definition (R&E 2013, p. 244):** Knowledge or intellectual property moves from one actor to another via market or non-market mechanisms. Core activity of the innovation system.

**Assign when the sentence describes:**
- Licensing of patents, know-how, or inventions
- Transfer of research results or IP for commercial application
- Technology transfer offices, science parks, incubators, spin-off accelerators serving one actor
- Spin-off creation, patenting, or commercialisation involving both actors
- One actor funding or supporting the R&D output of the other for application

**NLI templates this matches:**
> "{subj} licenses patents, know-how, or inventions for use by {obj}."
> "{subj} transfers research results or intellectual property to {obj} for commercial application."
> "{subj} operates technology transfer offices, incubators, or science parks that serve {obj}."
> "{subj} supports spin-off creation, patenting, or licensing activities involving {obj}."

**Signal words:** transfer, license, commercialise, spin-off, IP, patent, incubator, science park, technology transfer office

---

### `collaboration_conflict_moderation`
**Definition (R&E 2013, p. 245):** One actor mediates or helps resolve active tension or conflict of interest between institutional spheres. Specific to triadic systems — a third party turns conflict into convergence.

**Assign when the sentence describes:**
- Explicit mention of tensions, competing interests, or conflicts being resolved
- One actor negotiating or mediating between the other and a third party
- Building partnership structures that transform opposing interests into collaboration
- Conflict of interest between academia, industry, and government being addressed

**NLI templates this matches:**
> "{subj} helps resolve tensions or conflicts of interest between institutional spheres involving {obj}."
> "{subj} develops a partnership structure with {obj} to transform competing interests into collaboration."

**Signal words:** conflict, tension, competing interests, mediate, resolve, bridge differences, reconcile

**Note:** Do NOT assign for general cooperation or working together — that is `networking`. This label requires explicit conflict or tension being addressed.

---

### `collaborative_leadership`
**Definition (R&E 2013, p. 246):** One actor takes an asymmetric leadership role, convening or organizing other spheres into a shared agenda. The "innovation organizer" role — bringing actors together, building consensus, coordinating top-down and bottom-up.

**Assign when the sentence describes:**
- One actor explicitly leading, convening, or organizing the other into a joint initiative
- An innovation organizer role (an individual or institution that bridges spheres)
- One actor mobilizing, coordinating, or driving a shared platform or agenda involving the other
- Cross-institutional entrepreneurship where one actor takes the lead

**NLI templates this matches:**
> "{subj} acts as an innovation organizer, convening {obj} and other institutional spheres into a shared agenda."
> "{subj} leads and directly coordinates the activities of {obj} within a cross-sector initiative."
> "{subj} has mobilized or activated {obj} to participate in a jointly driven programme."
> "{subj} steers {obj} and other actors toward a shared strategic goal."

**Signal words:** lead, coordinate, convene, organise, mobilise, drive, steer, innovation organizer, bring together, align

**Distinction from `networking`:** Networking is symmetric (both actors form a link). Collaborative leadership is asymmetric — one actor clearly leads or organizes the other.

---

### `substitution`
**Definition (R&E 2013, p. 246):** One sphere fills a role that normally belongs to another sphere, because that other sphere is absent or weak.

**Assign when the sentence describes:**
- Government providing venture capital (normally industry's role)
- University doing firm formation or funding (normally industry's role)
- Industry providing education or training (normally university's role)
- Any actor explicitly stepping in to fill a gap left by the other's absence or weakness

**NLI templates this matches:**
> "{subj} fills a gap left by the absence or weakness of {obj} in the innovation system."
> "{subj} takes over a function normally belonging to {obj} because {obj} is absent or underdeveloped."
> "{subj} provides venture capital, firm formation, or training because {obj} lacks the capacity to do so."

**Signal words:** fill the gap, step in, take over, substitute, lack of, absence of, weak, underdeveloped, instead of, in place of

**Note:** This is rare in policy documents. Assign only when there is clear language about one actor filling another's absent or weak role — not just when they cooperate.

---

### `networking`
**Definition (R&E 2013, p. 246):** Symmetric connection-building between actors — forming alliances, consortia, bilateral or multilateral ties — without hierarchical dominance or conflict resolution.

**Assign when the sentence describes:**
- Both actors participating jointly in a network, consortium, platform, or programme
- Establishing ongoing linkages at regional, national, or international level
- Bilateral or multilateral agreements or partnerships (no one actor leads)
- General collaboration or working together without a specific leadership or transfer dynamic

**NLI templates this matches:**
> "{subj} and {obj} are joint members of a consortium, platform, or research network."
> "{subj} has established a formal partnership or bilateral agreement with {obj}."
> "{subj} and {obj} collaborate through an established joint programme or multilateral initiative."
> "{subj} and {obj} have built shared networks or innovation alliances at regional, national, or international level."

**Signal words:** network, consortium, partnership, collaborate, joint, together, alliance, platform, link, connect, cooperate

---

### `no_explicit_relation`
**Assign when:**
- The two actors are simply mentioned together in the same sentence without any specific relational dynamic
- The sentence lists multiple actors without describing how they interact
- The relation between them is only implied or too vague to classify
- The sentence is about policy aspirations or goals rather than a described interaction

**Examples of no_explicit_relation:**
> "Universities, industry and government all play a role in quantum research."
> "The strategy involves IBM, national laboratories, and universities."
> "Both the Ministry of Economy and universities support the initiative."

---

## Decision procedure

1. Read `sent_text` carefully.
2. Identify which two actors are `entity_1` and `entity_2`.
3. Ask: does the sentence describe a specific interaction between them?
   - No → `no_explicit_relation`
   - Yes → which type?
4. Apply the label that best fits the **dominant** dynamic in the sentence.
5. If two labels seem equally applicable, prefer the more specific one:
   - `substitution` > `technology_transfer` > `collaborative_leadership` > `networking`
   - `collaboration_conflict_moderation` only when conflict is explicitly mentioned

## Valid labels
```
technology_transfer
collaboration_conflict_moderation
collaborative_leadership
substitution
networking
no_explicit_relation
```
