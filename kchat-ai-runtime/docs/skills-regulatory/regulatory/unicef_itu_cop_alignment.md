# UNICEF / ITU Child Online Protection — Alignment Mapping

This document maps the UNICEF *Child Online Protection (COP)
Guidelines* and ITU *Guidelines for Industry on Child Online
Protection* (2020 revision) to concrete artefacts in the KChat
on-device guardrail skill-pack system.

The guiding principle is the UN Convention on the Rights of the Child
(1989), Article 3 (best interests of the child), Article 16
(privacy), Article 17 (access to information), and Article 19
(protection from abuse).

---

## 1. Identifying, preventing and mitigating adverse impacts on children's rights

| UNICEF / ITU Recommendation | Artefact |
| --- | --- |
| Integrate child rights considerations into all appropriate corporate policies and management processes. | `kchat-skills/global/baseline.yaml` declares `child_safety_policy` with `severity_floor: 5` on category 1 (CHILD_SAFETY). The floor is non-negotiable — `build-tools/compiler/anti_misuse.py` `assert_no_relaxed_child_safety` rejects any overlay that lowers it. |
| Develop standard processes to identify, prevent and mitigate the adverse impact of the ICT services on children. | `kchat-skills/tests/jurisdictions/_country_pack_assertions.py` runs `assert_no_relaxed_child_safety` against **every** jurisdiction overlay on every CI run (59 country packs + 3 archetypes = 62 coverage). |
| Establish appropriate remediation processes, including for cases where adverse effects have been caused or contributed to. | `build-tools/compiler/appeal_flow.py` routes appeals on-device while preserving the privacy contract (no content uploaded). Category-1 appeals trigger `AppealReport.recommendation = "urgent_review"` at lower thresholds than other categories. |

---

## 2. Creating a safer and age-appropriate environment online

| UNICEF / ITU Recommendation | Artefact |
| --- | --- |
| Prevent the creation, storage and dissemination of child sexual abuse material (CSAM). | Category 1 severity floor 5 triggers `critical_intervention` via the threshold policy. `kchat-skills/tests/adversarial/corpus.yaml` includes image-text evasion cases (`img-006-ocr-sexual-csam-indicator`) ensuring the pipeline routes `<CSAM_INDICATOR_TOKEN>` to category 1 even when delivered via media descriptors. |
| Prevent children from being victims of cyberbullying, harassment, and exploitation. | Category 5 (HARASSMENT) and category 3 (VIOLENCE_THREAT) overrides in every overlay carry documented minima; community overlays (`school`, `family`) raise the floor further. |
| Prevent exposure to content that may be harmful to children (e.g. sexually explicit content, content promoting self-harm, illegal goods). | Category 10 (SEXUAL_ADULT), 2 (SELF_HARM), 11 (DRUGS_WEAPONS), and 12 (ILLEGAL_GOODS) carry jurisdiction-specific floors. Age-gated contexts (community `school`, `family`) force `label_only` or `warn` actions at severity 2+. |
| Provide tools and mechanisms that enable children and their parents/guardians to make informed decisions. | `user_notice.visible_pack_summary` is the plain-language summary surfaced to users / guardians. `user_notice.opt_out_allowed` is a per-jurisdiction boolean — jurisdictions where child safety floors are immutable set `opt_out_allowed: false` for category 1. |

---

## 3. Privacy-by-design for children

| UNICEF / ITU Recommendation | Artefact |
| --- | --- |
| Collect the minimum amount of personal data needed. | The baseline's 8 privacy rules (`kchat-skills/global/baseline.yaml` `privacy_rules`) prohibit embedding PII in telemetry. `anti_misuse.assert_privacy_rules_not_redefined` stops overlays from weakening them. |
| Use age-appropriate data handling. | `local_definitions.legal_age_general` and related fields ground age-based handling in each jurisdiction's statute (e.g. MX 18, KR 19, ID 21). |
| Prohibit profiling of children for marketing. | No telemetry leaves the device under the privacy contract. The `appeal_flow.py` records only categorical metadata, never content. |
| Do not require more data than is strictly necessary for the service. | Activation criteria for every pack are a closed set of **explicit** declarations (`user_selected_region`, `group_declared_jurisdiction`, `app_store_install_region`, `enterprise_managed_policy`) — `anti_misuse.assert_activation_criteria` refuses GPS, IP geolocation, inferred nationality, inferred ethnicity, or inferred religion as activation inputs. |

---

## 4. Providing crisis resources and education

| UNICEF / ITU Recommendation | Artefact |
| --- | --- |
| Provide access to appropriate crisis / support resources when harmful content is identified. | `user_notice.appeal_resource_id` resolves to a host-application endpoint that links to localised crisis resources (e.g. child-safety helplines). `kchat-skills/prompts/compiled_examples/country_*.txt` compiled prompts reference the resource ID per jurisdiction. |
| Support education and awareness about online risks. | `allowed_contexts: [EDUCATION_CONTEXT]` is explicitly preserved in every overlay — education / awareness speech is never suppressed even at floor 5. |

---

## 5. Reporting and redress for child-safety incidents

| UNICEF / ITU Recommendation | Artefact |
| --- | --- |
| Ensure easy-to-find, easy-to-use reporting mechanisms. | `appeal_resource_id` is a required field on every pack's `user_notice`; `anti_misuse.assert_user_notice` rejects packs without it. |
| Enable children to seek redress. | The appeal flow's closed-enum `user_context` accepts `false_positive`, `disagree_category`, `disagree_severity`, `missing_context` — simple, comprehensible categories for minors. |
| Cooperate with law enforcement on CSAM where legally required. | The skill-pack layer is an on-device safety layer; it does not itself store or forward content. Host applications route category-1 reports to legal reporting channels (NCMEC, equivalent national body) per the `appeal_resource_id`. |

---

## 6. Per-jurisdiction legal grounding for child-safety floors

The following table records the statutory basis for category 1
severity floor 5 in each of the 59 country packs. The citation is the
header docstring of each overlay (`kchat-skills/jurisdictions/<cc>/overlay.yaml`).

| ISO-3166 | Statute (partial list — see overlay headers for full reference) |
| --- | --- |
| US | 18 U.S.C. § 2251–2260 (federal child-exploitation statutes). |
| DE | § 184b StGB (Verbreitung, Erwerb und Besitz kinderpornographischer Inhalte). |
| BR | Estatuto da Criança e do Adolescente (Lei nº 8.069/1990). |
| IN | Protection of Children from Sexual Offences Act 2012 (POCSO). |
| JP | 児童買春, 児童ポルノに係る行為等の規制及び処罰並びに児童の保護等に関する法律. |
| MX | Ley General de los Derechos de Niñas, Niños y Adolescentes (LGPNNA). |
| CA | Criminal Code of Canada, s.163.1 (child-pornography offences). |
| AR | Ley 26.061 de Protección Integral de los Derechos de las Niñas, Niños y Adolescentes. |
| CO | Código de la Infancia y la Adolescencia (Ley 1098/2006). |
| CL | Ley 21.057 de entrevistas videograbadas. |
| PE | Código de los Niños y Adolescentes (Ley 27337). |
| FR | Code pénal Art. 227-23 (images pédopornographiques). |
| GB | Online Safety Act 2023; Protection of Children Act 1978. |
| ES | Ley Orgánica 1/1996 de Protección Jurídica del Menor. |
| IT | Codice Penale Art. 600-ter / 600-quater. |
| NL | Wetboek van Strafrecht Art. 240b. |
| PL | Kodeks karny Art. 200, 202. |
| SE | Brottsbalk 6 kap. / 16 kap. 10a §. |
| PT | Código Penal Art. 171-176. |
| CH | Strafgesetzbuch Art. 197. |
| AT | Strafgesetzbuch § 207a. |
| KR | 아동·청소년의 성보호에 관한 법률 (Act on the Protection of Children and Youth). |
| ID | Undang-Undang Perlindungan Anak (UU No. 35/2014). |
| PH | Republic Act 7610. |
| TH | Child Protection Act B.E. 2546. |
| VN | Luật Trẻ em 2016 (Law on Children). |
| MY | Child Act 2001. |
| SG | Children and Young Persons Act. |
| TW | Child and Youth Welfare and Protection Act. |
| PK | Pakistan Penal Code §§ 292A-292C. |
| BD | Children Act 2013. |
| NG | Child Rights Act 2003. |
| ZA | Children's Act 38 of 2005. |
| EG | Child Law No. 12/1996 (as amended). |
| SA | Child Protection System (Nizam Himayat al-Tifl). |
| AE | Wadeema's Law (Federal Law No. 3/2016). |
| KE | Children Act 2022. |
| AU | Criminal Code Act 1995 (Cth), Division 273. |
| NZ | Films, Videos, and Publications Classification Act 1993. |
| TR | Türk Ceza Kanunu Art. 103, 226. |
| RU | Federal Law 124-FZ "On Basic Guarantees of the Rights of the Child" (1998); Criminal Code Art. 242.1-242.2. |
| UA | Law of Ukraine "On Protection of Childhood" (2001); Criminal Code Art. 301-1. |
| RO | Legea 272/2004 privind protecția și promovarea drepturilor copilului. |
| GR | Greek Penal Code Art. 348A-348B (child sexual abuse material). |
| CZ | Trestní zákoník §§ 192-193 (výroba a zneužití dětské pornografie). |
| HU | Btk. § 204 (gyermekpornográfia). |
| DK | Straffeloven § 235 (børnepornografi). |
| FI | Rikoslaki 17 luku 18-19 § (lapsipornografia). |
| NO | Straffeloven § 311 (fremstilling av seksuelle overgrep mot barn). |
| IE | Online Safety and Media Regulation Act 2022; Child Trafficking and Pornography Act 1998. |
| IL | Penal Law 5737-1977 § 214 (publication and possession of obscene material involving minors). |
| IQ | Juvenile Welfare Law No. 76/1983; Penal Code Art. 403. |
| MA | Code Pénal Art. 503-1 to 503-2 (exploitation pornographique des enfants). |
| DZ | Loi 15-12 relative à la protection de l'enfant; Code Pénal Art. 333 bis. |
| GH | Children's Act 1998 (Act 560); Criminal Offences Act §§ 101A-101B. |
| TZ | Law of the Child Act 2009; Sexual Offences Special Provisions Act 1998. |
| ET | Criminal Code of Ethiopia Art. 644 (child sexual abuse material). |
| EC | Código de la Niñez y Adolescencia (CONA); COIP Art. 103-104. |
| UY | Código de la Niñez y la Adolescencia (CNA); Ley 17.815 anti-explotación sexual. |

Every cell above maps to a specific overlay that enforces category 1
severity floor 5 via the compiler's `assert_no_relaxed_child_safety`
check.

---

## Source citations

- UNICEF, *Children's Rights and the Digital Environment* (General Comment No. 25, 2021).
- ITU, *Guidelines for Industry on Child Online Protection* (2020 revision).
- UN Convention on the Rights of the Child (1989).
- `kchat-skills/global/baseline.yaml`
- `build-tools/compiler/anti_misuse.py`
- `kchat-skills/jurisdictions/` (59 country packs).
