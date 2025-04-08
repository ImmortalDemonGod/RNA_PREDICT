# rna_predict/pipeline/stageD/diffusion/components/diffusion_schedule.py
import torch


class DiffusionSchedule:
    def __init__(
        self,
        sigma_data: float = 16.0,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        p: float = 7.0,
        dt: float = 1 / 200,
        p_mean: float = -1.2,
        p_std: float = 1.5,
    ) -> None:
        """
        Args:
            sigma_data (float, optional): The standard deviation of the data. Defaults to 16.0.
            s_max (float, optional): The maximum noise level. Defaults to 160.0.
            s_min (float, optional): The minimum noise level. Defaults to 4e-4.
            p (float, optional): The exponent for the noise schedule. Defaults to 7.0.
            dt (float, optional): The time step size. Defaults to 1/200.
            p_mean (float, optional): The mean of the log-normal distribution for noise level sampling. Defaults to -1.2.
            p_std (float, optional): The standard deviation of the log-normal distribution for noise level sampling. Defaults to 1.5.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.p = p
        self.dt = dt
        self.p_mean = p_mean
        self.p_std = p_std
        # self.T
        self.T = int(1 / dt) + 1  # 201

    def get_train_noise_schedule(self) -> torch.Tensor:
        return self.sigma_data * torch.exp(self.p_mean + self.p_std * torch.randn(1))

    def get_inference_noise_schedule(self) -> torch.Tensor:
        time_step_lists = torch.arange(start=0, end=1 + 1e-10, step=self.dt)
        inference_noise_schedule = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.p)
                + time_step_lists
                * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )
        return inference_noise_schedule
