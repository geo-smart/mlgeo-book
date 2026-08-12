# Course Schedule — Autumn 2026

ESS 469/569, University of Washington. Class meets **Monday, Wednesday, Friday, 10:00–11:20 in SIG 227** (Sieg Hall). Bring a charged laptop — the room has tablet-arm seating and limited power; lab sessions run the full 80 minutes. Instruction runs September 30 – December 11, 2026; there is no class on **Wednesday November 11** (Veterans Day) or **Friday November 27** (Native American Heritage Day), and Thanksgiving is Thursday November 26. Final examination week is December 12–18.

This page is the course-delivery layer on top of the book: which sessions cover which sections, and when graded work is due. The book carries the full depth; in-class delivery selects from it (see the 469/569 differentiation notes in each assignment).

Two design choices shape the quarter. First, **the agent thread runs through the whole course** rather than waiting for the end: what agents are and how we use them (week 2), turning your data skills on an AI's claims (week 4), and building eval sets for agents (week 7) — so the course's distinctive material is taught early and applied often, and the capstone gets a month of runway. Second, **December contains no new core material**: only application, clinics, one deliberate buffer session, and presentations, so nothing important is lost if a week slips.

**The paper-pulse rhythm.** From week 3 through week 10, most sessions open with two 4-minute student talks — one paper from the speaker's [reading-arc](../Chapter6-AgenticAI/6.5_reading_arc.md) literature review, dissected against the class's rubric-in-progress. Every student presents once (sign-ups open Mon Oct 5 on Canvas) and everyone in the room scores every talk through the standing peer-feedback survey: five rubric ratings plus one strength and one improvement, anonymized to the presenter, completion counted as participation. Pulse talks skip the two 4.5 lab days and check-in #1; the same survey instrument returns for the final presentations.

## Graded-work calendar at a glance

| Item | Opens | Due / window |
|---|---|---|
| HW1 — workbench setup ([1.9](../Chapter1-GettingStarted/1.9_workbench_setup_hw1.md)) | Sep 30 | Mon Oct 12 |
| Ch 1 quiz (Canvas, timed) | Tue Oct 6 | Thu Oct 8 |
| Reading arc stage 1 — AI-assisted lit review ([6.5](../Chapter6-AgenticAI/6.5_reading_arc.md)) | Oct 5 | Wed Oct 21 |
| Ch 2 quiz | Mon Oct 26 | Wed Oct 28 |
| Final project proposal ([1.10](1.10_MLGEO_FinalProject.md)) | Oct 12 | Fri Oct 30 |
| Classification leaderboard ([3.5](../Chapter3-MachineLearning/3.5_multiclass_classification.ipynb)) | Mon Nov 2 | closes Tue Nov 24 |
| Reading arc stage 2 — anatomy of good papers | Oct 21 | Wed Nov 4 |
| Ch 3 quiz (includes the flipped 3.10 reading) | Tue Nov 10 | Thu Nov 12 |
| Reading arc stage 3 — your quality rubric | Nov 4 | Fri Nov 13 |
| Project check-in #1 — data-audit studio | — | Mon Nov 16, in class |
| Ch 6 quiz | Mon Nov 16 | Wed Nov 18 |
| HW-CML ([Homework_CML](../Chapter3-MachineLearning/Homework_CML.ipynb)) | Nov 4 | Fri Nov 20 |
| HW-DL ([Homework_DL](../Chapter4-DeepLearning/Homework_DL.ipynb)) | Nov 23 | Fri Dec 4 |
| Ch 5 quiz (checks the flipped Ch 5 reading) | Mon Nov 30 | Thu Dec 3 |
| Ch 4 quiz (after the forecasting session) | Thu Dec 3 | Mon Dec 7 |
| Forecasting leaderboard ([4.10](../Chapter4-DeepLearning/mlgeo_4.10_timeseriesforecast.ipynb)) | Wed Dec 2 | closes Wed Dec 9 |
| Project check-in #2 — dry-runs + agent clinic | — | Mon Dec 7, in class |
| Reading arc stage 4 — your pre-submission review agent | Nov 13 | Thu Dec 10 |
| Final presentations | — | Fri Dec 11 in class + assigned finals slot |
| Final report + repository | — | Wed Dec 16 |

Chapter 7 has no quiz: its outcomes (audience translation, downstream impact) are assessed directly by the final-project deliverables. Quizzes are timed, auto-graded on Canvas, open for the window listed, and drawn from scenario banks — they test judgment about situations, not recall.

## Week by week

**Week 1 — Open, reproducible science** (2 sessions)
- Wed Sep 30 — Course introduction; why ML in the geosciences; open reproducible science ([1.1](../Chapter1-GettingStarted/1.1_open_reproducible_science.md)). HW1 assigned; setup help continues in the week-1 install clinic (office hours), not in lecture.
- Fri Oct 2 — Workbench lab: version control and environments ([1.2](../Chapter1-GettingStarted/1.2_jupyter_environment.md)–[1.5](../Chapter1-GettingStarted/1.5_version_control_git.md), with [1.9](../Chapter1-GettingStarted/1.9_workbench_setup_hw1.md) as the self-serve walkthrough); the pull-request dry run.

**Week 2 — Agents, then data** (Ch 1 quiz Tue–Thu)
- Mon Oct 5 — **Working with agents: policy and mechanism** ([1.8](../Chapter1-GettingStarted/1.8_ai_in_your_workflow.md) + [6.1](../Chapter6-AgenticAI/6.1_llms_to_agents.md)): what an agent is, where errors enter the loop, the badge system, disclosure. Reading arc stage 1 assigned (topic: opportunities and challenges of AI in your subfield, with the two anchor readings); paper-pulse sign-ups open.
- Wed Oct 7 — Meet the data: definitions and formats ([2.1](../Chapter2-DataManipulation/2.1_Data_Definitions.md)–[2.2](../Chapter2-DataManipulation/2.2_data_formats_rendered.ipynb)), the data gallery, and a first real dataset ([1.6](../Chapter1-GettingStarted/1.6_data_gallery.md)–[1.7](../Chapter1-GettingStarted/1.7_get_geodetic_gnss.ipynb)).
- Fri Oct 9 — Tables: pandas and dataframe preparation ([2.3](../Chapter2-DataManipulation/2.3_pandas_rendered.ipynb)–[2.4](../Chapter2-DataManipulation/2.4_dataframes_prep.ipynb)).

**Week 3 — Signals** (HW1 due Mon)
- Mon Oct 12 — Arrays and gridded data ([2.5](../Chapter2-DataManipulation/2.5_Arrays.ipynb)); resampling and irregular data ([2.6](../Chapter2-DataManipulation/2.6_resampling.ipynb)).
- Wed Oct 14 — Statistical considerations ([2.7](../Chapter2-DataManipulation/2.7_statistical_considerations.ipynb)); spectral transforms ([2.8](../Chapter2-DataManipulation/2.8_data_spectral_transforms.ipynb)).
- Fri Oct 16 — Filtering, gaps, timing errors: repairing real records ([2.9](../Chapter2-DataManipulation/2.9_filtering_data.ipynb)).

**Week 4 — AI-ready data, then interrogate the AI** (arc stage 1 due Wed)
- Mon Oct 19 — Synthetic data and the STA/LTA detection floor ([2.10](../Chapter2-DataManipulation/2.10_synthetic_noise.ipynb)); feature engineering ([2.11](../Chapter2-DataManipulation/2.11_feature_engineering.ipynb)).
- Wed Oct 21 — Dimensionality reduction ([2.12](../Chapter2-DataManipulation/2.12_dimensionality_reduction.ipynb)); the AI-ready checklist and raster-to-station joins ([2.13](../Chapter2-DataManipulation/2.13_MLready_data.ipynb)).
- Fri Oct 23 — **Critical-evaluation lab** ([6.2](../Chapter6-AgenticAI/6.2_critical_evaluation.ipynb)) — Chapter 2's capstone: turn your new data skills on an AI's claims and verify against the data, not against plausibility.

**Week 5 — Classic ML begins** (Ch 2 quiz Mon–Wed; proposals due Fri)
- Mon Oct 26 — Supervision concepts; classification vs regression ([3.1](../Chapter3-MachineLearning/3.1_concepts_supervision.md)–[3.2](../Chapter3-MachineLearning/3.2_classification_regression.ipynb)).
- Wed Oct 28 — Clustering ([3.3](../Chapter3-MachineLearning/3.3_clustering.ipynb)).
- Fri Oct 30 — Binary classification and imbalance ([3.4](../Chapter3-MachineLearning/3.4_binary_classification.ipynb)). **Project proposals due.**

**Week 6 — Classification, honestly** (arc stage 2 due Wed)
- Mon Nov 2 — Multiclass classification ([3.5](../Chapter3-MachineLearning/3.5_multiclass_classification.ipynb)); how the leaderboard and hidden test sets work. **Classification leaderboard opens.**
- Wed Nov 4 — Logistic regression from scratch; are your probabilities honest? ([3.6](../Chapter3-MachineLearning/3.6_logistic_regression.ipynb)). HW-CML assigned.
- Fri Nov 6 — Trees, forests, ensembles ([3.7](../Chapter3-MachineLearning/3.7_randomForest_regression.ipynb) + [3.9](../Chapter3-MachineLearning/3.9_ensemble_learning.ipynb)); reading importances without fooling yourself. [3.10](../Chapter3-MachineLearning/3.10_autoML.ipynb) assigned as flipped reading (checked by the Ch 3 quiz).

**Week 7 — The two evaluation lectures** (short week — no class Wed Nov 11; Ch 3 quiz Tue–Thu; arc stage 3 due Fri)
- Mon Nov 9 — **Robust training** ([3.8](../Chapter3-MachineLearning/3.8_robust_training.ipynb)), full session: why random splits lie; temporal, grouped, and spatial splits; the split ladder on data with planted truth.
- Fri Nov 13 — **Build an eval set** ([6.3](../Chapter6-AgenticAI/6.3_build_an_eval_set.ipynb)) + disclosure and norms ([6.4](../Chapter6-AgenticAI/6.4_disclosure_and_norms.md)): the same fair-evaluation move, applied to agents. Reading arc stage 4 assigned — four weeks of runway.

**Week 8 — Check-in, then deep learning** (Ch 6 quiz Mon–Wed; HW-CML due Fri)
- Mon Nov 16 — **Project check-in #1: data-audit studio.** Each team, five minutes: your data against the 2.13 checklist, your baseline, your split design — fresh from 3.8.
- Wed Nov 18 — Perceptrons to MLPs ([4.0](../Chapter4-DeepLearning/mlgeo_4.0_perceptrons.ipynb)–[4.2](../Chapter4-DeepLearning/mlgeo_4.2_MLP.ipynb)).
- Fri Nov 20 — CNNs: images, waveforms, and the classical baseline ([4.3](../Chapter4-DeepLearning/mlgeo_4.3_CNN.ipynb)).

**Week 9 — Sequence models and the training lab** (short week — Thanksgiving)
- Mon Nov 23 — Sequence models: RNN, LSTM, attention ([4.4](../Chapter4-DeepLearning/mlgeo_4.4_RNN.ipynb)). HW-DL assigned.
- Wed Nov 25 — Model training lab, session I: data curation and label quality ([4.5](../Chapter4-DeepLearning/mlgeo_4.5_ModelTraining.ipynb) sections 1–3).

**Week 10 — Uncertainty, forecasting, flipped workflows** (Ch 5 quiz Mon–Thu; classification leaderboard closed Nov 24)
- Mon Nov 30 — Model training lab, session II: architecture, uncertainty, calibration, out-of-range behavior (4.5 sections 4–5). Ch 5 ([5.1](../Chapter5-ModelWorkflows/5.1_reproducibility.md)–[5.5](../Chapter5-ModelWorkflows/5.5_data_at_scale.ipynb)) assigned as flipped reading.
- Wed Dec 2 — Time-series forecasting shootout, probabilistic forecasts, skill horizons ([4.10](../Chapter4-DeepLearning/mlgeo_4.10_timeseriesforecast.ipynb)). **Forecasting leaderboard opens.** Autoencoders and PINNs ([4.6](../Chapter4-DeepLearning/mlgeo_4.6_AutoEncoder.ipynb)–[4.7](../Chapter4-DeepLearning/mlgeo_4.7_PINN.ipynb)) as enrichment (569 required, 469 optional).
- Fri Dec 4 — Ch 5 discussion (30 min, from the flipped reading) + communicating your science: audiences and downstream impact ([7.1](../Chapter7-UseCases/7.1_audience_translation.md)–[7.2](../Chapter7-UseCases/7.2_downstream_impact.md)). **HW-DL due.**

**Week 11 — Clinics, buffer, presentations** (Ch 4 quiz closes Mon; forecasting leaderboard closes Wed; arc stage 4 due Thu)
- Mon Dec 7 — **Project check-in #2: presentation dry-runs + review-agent clinic.** Rehearse; run your stage-4 agent on another team's draft.
- Wed Dec 9 — **Buffer session.** Absorbs any slippage from the quarter; if nothing slipped, a synthesis discussion of the quarter's question — opportunities and challenges of AI for geoscience, argued from the class's own measurements (agent evals, peer-score agreement statistics) — or a deep dive by class vote (live agent-eval demo, PINNs, data-at-scale).
- Fri Dec 11 — **Final presentations, session I.**

**Finals week** — Presentations session II in the registrar-assigned exam slot (check the Time Schedule); **final report and repository due Wednesday December 16**.

## Notes on pacing

- The agent thread is deliberately early: verification culture (6.1/6.2) is installed before homework habits form, 6.3 lands the week after 3.8 so fair evaluation reads as one idea applied twice, and stage 4 of the [reading arc](../Chapter6-AgenticAI/6.5_reading_arc.md) has a month of runway. The Ch 6 quiz follows 6.3/6.4 in mid-November.
- Notebook 4.5 is split across two lab sessions at its marked checkpoint; do not attempt it in one sitting.
- Flipped chapters (3.10, Ch 5) are examinable through their quizzes; the in-class time they would have used funds the check-ins and the buffer.
- Check-in #1 replaces a written midterm report: earlier feedback, at the moment it can still change the project.
- 469 students follow the diluted path noted in each assignment: Apply-level expectations on the uncertainty and agent outcomes, 4.6/4.7 optional, and assisting rather than leading final projects.
- Sections not lectured are not optional reading for 569 unless marked so; lectures select, the book carries the depth.
