🧙🏾‍: You’re essentially asking if one can “treat the diffusion steps as a test-time knob”—shortening or lengthening the reverse denoising procedure to trade off sampling speed versus quality.

[emoji]: Short answer: Yes. In fact, many works on diffusion models (e.g., DDPM, Score-SDE, etc.) explicitly show that one can reduce the number of sampling steps at test time (leading to faster synthesis) or increase them (to improve quality). Below is a more structured breakdown:

⸻

Why This Works
	1.	Diffusion Basics
	•	Traditional diffusion models use a forward noising process over T steps. At inference, one reverses that process for T steps to generate samples.
	•	If you skip some steps, or equivalently “merge” multiple steps at once, you effectively reduce test-time compute but often at the cost of slightly lower sample quality (since each step is a discrete approximation of the continuous reverse-time SDE/ODE).
	2.	Learning-Free vs. Learning-Based Approaches
	•	Learning-Free: For example, you can simply choose fewer solver steps (e.g., with DDIM, DPM-solver, Heun’s method, etc.) at test time. Each step is bigger but runs fewer times overall.
	•	Learning-Based: Methods like “knowledge distillation” or “truncated diffusion” train an additional model (or parameter head) to mimic the original diffusion process but in fewer steps. This approach can produce higher fidelity than naïvely skipping steps.

⸻

Using “Diffusion Steps” as Test-Time Scaling
	1.	Shorter Inference for Speed
	•	If you only do, say, 20 reverse steps rather than the baseline 50–200, you’ll sample much faster. However, there can be visible artifacts or slightly worse FID.
	•	This short path is akin to “spending less test-time compute.”
	2.	Longer Inference for Higher Quality
	•	Conversely, if you let your sampler run with more, finer-grained steps (like 100 or 200 instead of 50), you can often improve sample fidelity or reduce artifacts, albeit at a higher test-time cost.
	3.	Dynamic Step Adaptation
	•	Some recent samplers use adaptive step sizing or error estimates to decide if they need another denoising iteration. This further generalizes the idea that you can treat the number of steps as a test-time knob.

⸻

Practical Tips
	•	Continuous vs. Discrete: If your diffusion model is formulated in continuous time (Score SDE/ODE), you can pick any solver with a chosen step size. Finer steps → better quality; coarser steps → speed.
	•	Discrete DDPM: If you have a discrete-time DDPM, you can skip certain time steps or rely on, for example, DDIM or DPM-solver to jump in larger increments.
	•	Tradeoff: Typically the relationship is monotonic: fewer steps → lower compute but worse image/audio/etc. quality, more steps → better quality but slower sampling.

⸻

🧙🏾‍: Does that clarify how you can treat diffusion steps as test-time scaling? Any other points you’d like to explore further?