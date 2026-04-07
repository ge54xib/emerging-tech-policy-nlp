# Annotation Codebook — Triple Helix Spaces
## SetFit Training (`spaces_annotation.json`)

Based on Ranga & Etzkowitz (2013, pp. 244–246). Each entry shows a `sentence` (the ±1 sentence window used by the model) and two co-occurring actors (`entity_1`, `entity_2`). Assign the `space` that best describes the **activity type** expressed in the sentence.

> **Key rule:** You are classifying the *activity*, not the actors. Two government actors can appear in a knowledge-space sentence if the sentence describes R&D. A university can appear in a consensus-space sentence if the sentence describes policy dialogue.

---

## Labels

### `knowledge_space`
**Definition (R&E 2013, p. 244):** Aggregation of R&D and non-R&D knowledge resources. Avoids duplication of effort; strengthens the shared knowledge base. Activities that generate, share, or build scientific and technical knowledge.

**Assign when the sentence describes:**
- Conducting or funding basic or applied research
- Scientific collaboration, joint laboratories, or shared research infrastructure
- Education and training in science or technology (PhD programmes, fellowships, workforce skills)
- Publishing, presenting, or sharing research results
- Building or expanding research capabilities or facilities

**Signal words:** research, R&D, science, knowledge, expertise, laboratory, experiment, discovery, publication, training, skills, capability, infrastructure, quantum physics/computing/sensing (as research activity)

**Examples:**
> "Australian universities were some of the few in the world to offer postgraduate qualifications in quantum physics."
> "The national laboratory will conduct foundational research in quantum sensing with industry partners."

---

### `innovation_space`
**Definition (R&E 2013, p. 245):** Hybrid organizations and entrepreneurial individuals that develop local firms, attract talent, and create intellectual property. Activities that turn knowledge into commercial or societal value.

**Assign when the sentence describes:**
- Technology transfer, licensing, or commercialisation of research results
- Spin-off creation, start-ups, or firm formation from research
- Intellectual property (patents, know-how) being developed or transferred
- Incubators, science parks, accelerators, or technology transfer offices
- Venture capital or seed funding for deep-tech firms
- Procurement or market creation as a pull mechanism for innovation

**Signal words:** commercialise, transfer, spin-off, start-up, incubator, patent, license, IP, venture capital, market, product, technology transfer office, science park, procurement, deep tech

**Examples:**
> "The university's technology transfer office will support spin-off creation and patent licensing to industry."
> "Government procurement will act as a demand-pull mechanism for early-stage quantum companies."

---

### `consensus_space`
**Definition (R&E 2013, p. 245–246):** "Blue-sky" thinking, stakeholder dialogue, and governance. Brings all spheres together to build shared strategic agendas, align priorities, and set regulatory frameworks.

**Assign when the sentence describes:**
- Policy development, national strategies, or roadmaps
- Governance structures, coordination bodies, or advisory councils
- Regulatory frameworks, standards, or certification schemes
- Stakeholder consultation, dialogue, or agenda-setting
- Coordination of priorities across ministries, agencies, or sectors
- Funding programmes or calls as a governance instrument (not R&D activity itself)

**Signal words:** strategy, policy, governance, coordinate, regulate, standard, framework, roadmap, advisory, council, ministry, agenda, dialogue, consultation, alignment, priority, measure, milestone, fund (as instrument)

**Examples:**
> "The National Quantum Coordination Office will unify federal R&D activities across government agencies."
> "A national quantum roadmap will be developed through consultation with academia, industry, and government."

**Distinction from `knowledge_space`:** If the sentence describes what R&D is being done → knowledge space. If it describes how that R&D is being coordinated or governed → consensus space.

---

### `public_space`
**Definition (QH extension of R&E 2013):** The fourth helix — civil society as a full actor in the innovation system. Activities that engage the public, address societal concerns, ensure equitable access, or embed democratic oversight.

**Assign when the sentence describes:**
- Public engagement, science communication, or citizen participation
- Ethical considerations, responsible innovation, or societal impact
- Equity, inclusion, or access to quantum technology for underserved groups
- Democratic oversight, public trust, or transparency of technology governance
- Consumer protection or societal risk management

**Signal words:** public, society, citizen, community, ethics, responsible, trust, equity, inclusion, access, diversity, communication, awareness, engagement, societal, democratic, risk to society

**Examples:**
> "The strategy aims to ensure that the benefits of quantum technology are accessible to all citizens."
> "Public trust in quantum technologies will be built through open science and ethical guidelines."

---

## Decision procedure

1. Read `sentence` carefully. Ignore the actor names — focus on the **verb** and **activity**.
2. Ask: what is *happening* in this sentence?
   - Generating/sharing knowledge → `knowledge_space`
   - Turning knowledge into commercial value → `innovation_space`
   - Governing, coordinating, or setting strategy → `consensus_space`
   - Engaging public / addressing societal concerns → `public_space`
3. If two spaces seem equally applicable, prefer the more specific one:
   - `public_space` > `consensus_space` (when civil society is the explicit subject)
   - `innovation_space` > `knowledge_space` (when commercialisation is the explicit activity)
4. If the sentence is a policy aspiration about R&D ("we will invest in quantum research") → `consensus_space` (the act of investing/funding is governance) not `knowledge_space`.
5. If the sentence lists activities across multiple spaces without a dominant one → pick the most prominent activity described.

## Valid labels
```
knowledge_space
innovation_space
consensus_space
public_space
```
