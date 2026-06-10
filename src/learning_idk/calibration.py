"""Temperature scaling and Expected Calibration Error (ECE).

Adapted from:
    Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger,
    "On Calibration of Modern Neural Networks," ICML 2017.
    https://arxiv.org/abs/1706.04599v2

Original source:
    https://github.com/osu-cvl/calibration/tree/main/temperature_scaling
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class IdentityNet(nn.Module):
    """Simple identity network used when only logits are provided."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class ModelWithTemperature(nn.Module):
    """A thin decorator which wraps a model with temperature scaling.

    Note: Output of the neural network should be the classification logits,
    NOT the softmax (or log softmax)!
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        n_bins: int = 15,
        strategy: str = "learn",
        per_class: bool = False,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        """
        Args:
            model: A torch nn.Module neural network. Defaults to identity network.
            n_bins: The number of bins used in ECE.
            strategy: The strategy used to temperature scale, either 'learn' or 'grid'.
            per_class: Perform temperature scaling per-class.
            device: The device to perform computations.
            verbose: Report updates on process.
        """
        super().__init__()
        if model is None:
            self.model = IdentityNet()
        else:
            self.model = model
        self.model.eval()
        self.model.to(device)
        self.strategy = strategy
        self.device = device
        self.per_class = per_class
        self.verbose = verbose
        self.n_bins = n_bins
        self.ece_criterion = ECE(n_bins=n_bins, device=device)
        self.temperature = torch.tensor(1.0, device=device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward function for nn.Module.

        Args:
            input: A tensor of inputs (rows are examples, columns are features).

        Returns:
            A tensor of temperature scaled logits.
        """
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Perform temperature scaling on logits.

        Args:
            logits: A tensor of logits (rows are examples, columns are features).

        Returns:
            A tensor of temperature scaled logits.
        """
        if logits.numel() == 0:
            return logits
        if not self.per_class:
            temperature = self.temperature.expand(logits.size(0), logits.size(1))
        else:
            preds = torch.argmax(logits, dim=1)
            temps = self.temperature.reshape(-1)
            if temps.numel() <= int(preds.max().item()):
                raise ValueError(
                    "Per-class temperature vector is smaller than the number of predicted "
                    f"classes. Got {temps.numel()} temperatures."
                )
            temperature = temps[preds].unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def global_temperature_scale(
        self, logits: torch.Tensor, temp: torch.Tensor
    ) -> torch.Tensor:
        """Perform temperature scaling on logits with a specified temperature.

        Args:
            logits: A tensor of logits (rows are examples, columns are features).
            temp: A scalar temperature value.

        Returns:
            A tensor of logits scaled using a scalar temperature.
        """
        return logits / temp.expand(logits.size(0), logits.size(1))

    def set_temperature(
        self,
        valid_loader: DataLoader,
        t_vals: list[float] | None = None,
        lrs: list[float] | None = None,
        num_iters: list[int] | None = None,
    ) -> torch.Tensor:
        """Resets and tunes the temperature of the model (using the validation set).

        Optimizes NLL for the learning approach and ECE for grid search.

        Args:
            valid_loader: Validation set loader.
            t_vals: Temperature values to search over (for 'grid' strategy).
            lrs: Learning rates for learned temperature scaling (for 'learn' strategy).
            num_iters: Max iterations for learned temperature scaling (for 'learn' strategy).

        Returns:
            Either a scalar float temperature, or a tensor of float temperatures.
        """
        if t_vals is None:
            t_vals = [1.0]
        if lrs is None:
            lrs = [0.01]
        if num_iters is None:
            num_iters = [50]

        self.nll_criterion = nn.CrossEntropyLoss().to(self.device)

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                labels_list.append(label.to(self.device))
                logits = self.model(input)
                logits_list.append(logits)
            if len(logits_list) == 0:
                raise ValueError("valid_loader is empty; cannot fit temperature scaling.")
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list)

        self.logits = logits
        self.targets = labels
        self.num_classes = logits.shape[1]

        if self.strategy == "grid":
            return self.set_temperature_grid(logits, labels, t_vals=t_vals)
        else:
            return self.set_temperature_learn(logits, labels, lrs=lrs, num_iters=num_iters)

    def set_temperature_learn(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
        lrs: list[float] | None = None,
        num_iters: list[int] | None = None,
    ) -> torch.Tensor:
        """Tune the temperature using learned temperature scaling (LBFGS on NLL).

        Args:
            all_logits: A tensor of all the logits from the validation loader.
            all_labels: A tensor of all the labels from the validation loader.
            lrs: Learning rates to search through.
            num_iters: Max iterations to search through.

        Returns:
            The best temperature or tensor of temperatures.
        """
        if lrs is None:
            lrs = [0.01]
        if num_iters is None:
            num_iters = [50]

        if self.per_class:
            optim_temps: list[torch.Tensor] = []
            preds = torch.argmax(all_logits, dim=1)
            for class_id in range(self.num_classes):
                if self.verbose:
                    print(f"Searching optimal temperature for class {class_id}")
                optim_temps.append(torch.tensor(1.0, device=self.device))
                logits = all_logits[preds == class_id]
                labels = all_labels[preds == class_id]
                if labels.shape[0] == 0:
                    continue
                before_temperature_nll = self.nll_criterion(logits, labels).item()
                before_temperature_ece = self.ece_criterion(logits, labels).item()
                if self.verbose:
                    print(
                        "Before temperature - NLL: %.3f, ECE: %.3f"
                        % (before_temperature_nll, before_temperature_ece)
                    )
                best_ece = before_temperature_ece
                best_nll = before_temperature_nll
                for lr in lrs:
                    for num_iter in num_iters:
                        temperature = nn.Parameter(
                            torch.tensor(1.0, device=self.device, requires_grad=True)
                        )

                        def closure():
                            optimizer.zero_grad()
                            loss = self.nll_criterion(
                                self.global_temperature_scale(logits, temperature), labels
                            )
                            loss.backward()
                            return loss

                        optimizer = optim.LBFGS([temperature], lr=lr, max_iter=num_iter)
                        optimizer.step(closure)

                        after_temperature_nll = self.nll_criterion(
                            self.global_temperature_scale(logits, temperature), labels
                        ).item()
                        after_temperature_ece = self.ece_criterion(
                            self.global_temperature_scale(logits, temperature), labels
                        ).item()
                        if after_temperature_ece < best_ece:
                            if self.verbose:
                                print(
                                    f"New Optimal temperature: {temperature.data.item():.3f}"
                                )
                            if self.verbose:
                                print(
                                    "After temperature - NLL: %.3f, ECE: %.3f"
                                    % (after_temperature_nll, after_temperature_ece)
                                )
                            optim_temps[class_id] = temperature.detach().clone()
                            best_ece = after_temperature_ece
                            best_nll = after_temperature_nll
                if self.verbose:
                    print(f"Optimal temperature: {optim_temps[class_id].item():.3f}")
                if self.verbose:
                    print(
                        "After temperature - NLL: %.3f, ECE: %.3f" % (best_nll, best_ece)
                    )

            self.temperature = torch.stack(optim_temps).to(self.device)

        else:
            optim_temp = torch.tensor(1.0, device=self.device)
            before_temperature_nll = self.nll_criterion(all_logits, all_labels).item()
            before_temperature_ece = self.ece_criterion(all_logits, all_labels).item()
            if self.verbose:
                print(
                    "Before temperature - NLL: %.3f, ECE: %.3f"
                    % (before_temperature_nll, before_temperature_ece)
                )
            best_ece = before_temperature_ece
            best_nll = before_temperature_nll
            for lr in lrs:
                for num_iter in num_iters:
                    temperature = nn.Parameter(
                        torch.tensor(1.0, device=self.device, requires_grad=True)
                    )

                    def closure():
                        optimizer.zero_grad()
                        loss = self.nll_criterion(
                            self.global_temperature_scale(all_logits, temperature),
                            all_labels,
                        )
                        loss.backward()
                        return loss

                    optimizer = optim.LBFGS([temperature], lr=lr, max_iter=num_iter)
                    optimizer.step(closure)

                    after_temperature_nll = self.nll_criterion(
                        self.global_temperature_scale(all_logits, temperature), all_labels
                    ).item()
                    after_temperature_ece = self.ece_criterion(
                        self.global_temperature_scale(all_logits, temperature), all_labels
                    ).item()
                    if after_temperature_ece < best_ece:
                        if self.verbose:
                            print(f"Optimal temperature: {temperature.item():.3f}")
                        optim_temp = temperature.detach().clone()
                        best_ece = after_temperature_ece
                        best_nll = after_temperature_nll

            self.temperature = optim_temp.to(self.device)
            if self.verbose:
                print(f"Optimal temperature: {optim_temp.item():.3f}")
            if self.verbose:
                print(
                    "After temperature - NLL: %.3f, ECE: %.3f" % (best_nll, best_ece)
                )

        return self.temperature

    def set_temperature_grid(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
        t_vals: list[float] | None = None,
    ) -> torch.Tensor:
        """Tune the temperature using grid search (minimizing ECE).

        Args:
            all_logits: A tensor of all the logits from the validation loader.
            all_labels: A tensor of all the labels from the validation loader.
            t_vals: Temperature values to search through.

        Returns:
            The best temperature or tensor of temperatures.
        """
        if t_vals is None:
            t_vals = [0.5, 1.0, 2.0]

        if self.per_class:
            preds = torch.argmax(all_logits, dim=1)
            optim_temps: list[torch.Tensor] = []
            for class_id in range(self.num_classes):
                optim_temps.append(torch.tensor(1.0, device=self.device))
                if self.verbose:
                    print(f"Searching optimal temperature for label class: {class_id}")

                c_idx = torch.where(preds == class_id)[0]
                logits = all_logits[c_idx]
                labels = all_labels[c_idx]

                if labels.shape[0] == 0:
                    continue

                before_temperature_nll = self.nll_criterion(logits, labels).item()
                before_temperature_ece = self.ece_criterion(logits, labels).item()
                if self.verbose:
                    print(
                        "\tBefore temperature - NLL: %.3f, ECE: %.3f"
                        % (before_temperature_nll, before_temperature_ece)
                    )
                best_ece = before_temperature_ece
                optim_temp = optim_temps[-1]

                for t in t_vals:
                    temp = torch.tensor(float(t), device=self.device)
                    if self.verbose:
                        print(f"\t\tTemperature values: {t}")
                    after_temperature_ece = self.ece_criterion(
                        self.global_temperature_scale(logits, temp), labels
                    ).item()
                    if self.verbose:
                        print(f"\t\tAfter temperature - ECE: {after_temperature_ece:.3f}")
                    if after_temperature_ece < best_ece:
                        best_ece = after_temperature_ece
                        optim_temp = temp
                        if self.verbose:
                            print(f"\t\tCurrent best ECE: {best_ece}")
                        if self.verbose:
                            print(f"\t\tCurrent optimum T: {optim_temp.item()}")

                if self.verbose:
                    print(f"\tFinal best ECE for label class {class_id}: {best_ece}")
                if self.verbose:
                    print(
                        f"\tFinal optimum T for label class {class_id}: {optim_temp.item()}"
                    )
                optim_temps[-1] = optim_temp

            self.temperature = torch.stack(optim_temps).to(self.device)
        else:
            if self.verbose:
                print("Searching optimal global temperature")
            logits = all_logits
            labels = all_labels
            before_temperature_ece = self.ece_criterion(logits, labels).item()
            if self.verbose:
                print(f"Before temperature - ECE: {before_temperature_ece:.3f}")
            best_ece = before_temperature_ece
            optim_temp = torch.tensor(1.0, device=self.device)
            for t in t_vals:
                if self.verbose:
                    print(f"\tTemperature values: {t}")
                temp = torch.tensor(float(t), device=self.device)
                after_temperature_ece = self.ece_criterion(
                    self.global_temperature_scale(logits, temp), labels
                ).item()
                if self.verbose:
                    print(f"\tAfter temperature - ECE: {after_temperature_ece:.3f}")
                if after_temperature_ece < best_ece:
                    best_ece = after_temperature_ece
                    optim_temp = temp
                    if self.verbose:
                        print(f"\tCurrent best ECE: {best_ece}")
                    if self.verbose:
                        print(f"\tCurrent optimum T: {optim_temp.item()}")
            if self.verbose:
                print(f"Final best ECE: {best_ece}")
            if self.verbose:
                print(f"Final optimum T: {optim_temp.item()}")
            self.temperature = optim_temp.to(self.device)

        return self.temperature

    def reliability_diagram_and_bin_count(self) -> None:
        """Plots reliability and bin count diagrams."""
        if self.per_class:
            preds = torch.argmax(self.logits, dim=1)
            for c_idx in range(self.num_classes):
                class_logits = self.logits[preds == c_idx]
                class_targets = self.targets[preds == c_idx]
                self.ece_criterion.reliability_diagram_and_bin_count(
                    logits=class_logits, targets=class_targets, title=f"Class-{c_idx}"
                )
        else:
            self.ece_criterion.reliability_diagram_and_bin_count(
                logits=self.logits, targets=self.targets
            )


class ECE(nn.Module):
    """Expected Calibration Error.

    Adapted from:
        https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

    The input to this loss is the logits of a model, NOT the softmax scores.
    Divides confidence outputs into equally-sized interval bins and computes
    a weighted average of |avg_confidence_in_bin - accuracy_in_bin|.

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI 2015.
    """

    def __init__(self, n_bins: int = 15, device: str = "cpu") -> None:
        """
        Args:
            n_bins: Number of confidence interval bins.
            device: Device for computations.
        """
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.n_bins = n_bins
        self.device = device

    def compute_ece(self, model: nn.Module, val_loader: DataLoader) -> torch.Tensor:
        """Compute ECE of a model on a data loader.

        Args:
            model: A model to compute ECE on.
            val_loader: A pytorch data loader.

        Returns:
            ECE on data loader.
        """
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                input = input.to(self.device)
                labels_list.append(label)
                logits = model(input)
                logits_list.append(logits)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)
        return self.forward(logits, labels)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, sm: bool = False
    ) -> torch.Tensor:
        if sm:
            self.sms = logits
        else:
            self.sms = torch.softmax(logits, dim=1)
        self.targets = labels

        confidences, predictions = torch.max(self.sms, 1)
        accuracies = predictions.eq(labels.int())
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def reliability_diagram_and_bin_count(
        self,
        logits: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        sm: bool = False,
        title: str = "",
    ) -> None:
        """Creates reliability diagram and bin count plots.

        Args:
            logits: A tensor of logits. If None, uses saved logits from forward pass.
            targets: A tensor of targets. If None, uses saved targets from forward pass.
            sm: If True, treat logits as softmax scores.
            title: A title to prepend to the default title.
        """
        if logits is not None:
            if sm:
                self.sms = logits
            else:
                self.sms = torch.softmax(logits, dim=1)
        if targets is not None:
            self.targets = targets

        bin_precision, count_in_bin = self.get_full_range_bin_precision()
        n_bin = len(bin_precision)
        bin_width = 1.0 / n_bin
        bin_center = torch.linspace(0.0 + 0.5 * bin_width, 1.0 + 0.5 * bin_width, n_bin + 1)[
            :-1
        ]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        if title != "":
            title = title + " "
        fig.suptitle(title + "Reliability Diagram and Bin Counts")
        ax0.bar(
            bin_center,
            bin_precision,
            align="center",
            width=bin_width * 0.7,
            label="Bin precision",
            color="orange",
        )
        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 1)
        ax0.plot(bin_center, bin_center, label="ideal case", color="blue", linestyle="-.")
        ax0.set_xlabel("Estimated label posterior")
        ax0.set_ylabel("Actual precision")
        ax0.legend()
        ax1.bar(
            bin_center,
            count_in_bin,
            align="center",
            width=bin_width * 0.7,
            label="Bin counts",
            color="blue",
        )
        for k, c in enumerate(count_in_bin):
            ax1.text(
                bin_center[k] - 0.005,
                count_in_bin[k] + 0.1,
                str(int(c)),
                color="black",
                fontsize="small",
                fontweight="bold",
            )
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Estimated label posterior")
        ax1.set_ylabel("Example counts in bin")
        ax1.legend()
        plt.show()

    def get_full_range_bin_precision(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conf, preds = torch.max(self.sms, dim=1)
        acc = preds == self.targets

        bin_precision = torch.zeros(self.n_bins)
        prop_in_bin = torch.zeros(self.n_bins)
        count_in_bin = torch.zeros(self.n_bins)
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = (conf >= bin_lower) * (conf <= bin_upper)
            prop_in_bin[i] = in_bin.float().mean()
            count_in_bin[i] = in_bin.sum()
            if prop_in_bin[i] > 0:
                bin_precision[i] = (1.0 * acc[in_bin]).mean()
        return bin_precision, count_in_bin


class PerClassECE(nn.Module):
    """Per-class Expected Calibration Error.

    Computes ECE separately for each predicted class and returns a tensor
    of per-class ECE values.
    """

    def __init__(self, n_bins: int = 15, device: str = "cpu") -> None:
        """
        Args:
            n_bins: Number of confidence interval bins.
            device: Device for computations.
        """
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.n_bins = n_bins
        self.device = device

    def compute_ece(self, model: nn.Module, val_loader: DataLoader) -> torch.Tensor:
        """Compute per-class ECE of a model on a data loader.

        Args:
            model: A model to compute per-class ECE on.
            val_loader: A pytorch data loader.

        Returns:
            Tensor of ECE scores per-class on data loader.
        """
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                input = input.to(self.device)
                labels_list.append(label)
                logits = model(input)
                logits_list.append(logits)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)
        return self.forward(logits, labels)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, sm: bool = False
    ) -> torch.Tensor:
        self.num_classes = logits.shape[1]
        if sm:
            self.sms = logits
        else:
            self.sms = torch.softmax(logits, dim=1)
        self.targets = labels
        confidences, predictions = torch.max(self.sms, dim=1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(self.num_classes).to(self.device)
        self.bin_accuracy = torch.zeros((self.num_classes, self.n_bins))
        self.prop_in_bin = torch.zeros((self.num_classes, self.n_bins))
        self.count_in_bin = torch.zeros((self.num_classes, self.n_bins))

        for c in range(self.num_classes):
            class_idx = torch.where(predictions == c)[0]
            class_confidences = confidences[class_idx]
            class_accuracies = accuracies[class_idx]

            for i, (bin_lower, bin_upper) in enumerate(
                zip(self.bin_lowers, self.bin_uppers)
            ):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(
                    bin_upper.item()
                )
                self.count_in_bin[c, i] = in_bin.sum()
                self.prop_in_bin[c, i] = in_bin.float().mean()
                if self.count_in_bin[c, i].item() > 0:
                    self.bin_accuracy[c, i] = class_accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].float().mean()
                    ece[c] += (
                        torch.abs(avg_confidence_in_bin - self.bin_accuracy[c, i])
                        * self.prop_in_bin[c, i]
                    )

        return ece

    def reliability_diagram_and_bin_count(
        self,
        logits: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        sm: bool = False,
    ) -> None:
        """Creates per-class reliability diagram and bin count plots.

        Args:
            logits: A tensor of logits. If None, uses saved logits from forward pass.
            targets: A tensor of targets. If None, uses saved targets from forward pass.
            sm: If True, treat logits as softmax scores.
        """
        if logits is not None:
            if sm:
                self.sms = logits
            else:
                self.sms = torch.softmax(logits, dim=1)
        if targets is not None:
            self.targets = targets
        self.num_classes = self.sms.shape[1]

        for c_idx in range(self.num_classes):
            bin_precision, count_in_bin = self.get_full_range_bin_precision(c_idx)
            n_bin = len(bin_precision)
            bin_width = 1.0 / n_bin
            bin_center = torch.linspace(
                0.0 + 0.5 * bin_width, 1.0 + 0.5 * bin_width, n_bin + 1
            )[:-1]
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
            title = f"Class-{c_idx} "
            fig.suptitle(title + "Reliability Diagram and Bin Counts")
            ax0.bar(
                bin_center,
                bin_precision,
                align="center",
                width=bin_width * 0.7,
                label="Bin precision",
                color="orange",
            )
            ax0.set_xlim(0, 1)
            ax0.set_ylim(0, 1)
            ax0.plot(
                bin_center, bin_center, label="ideal case", color="blue", linestyle="-."
            )
            ax0.set_xlabel("Estimated label posterior")
            ax0.set_ylabel("Actual precision")
            ax0.legend()
            ax1.bar(
                bin_center,
                count_in_bin,
                align="center",
                width=bin_width * 0.7,
                label="Bin counts",
                color="blue",
            )
            for k, c in enumerate(count_in_bin):
                ax1.text(
                    bin_center[k] - 0.005,
                    count_in_bin[k] + 0.1,
                    str(int(c)),
                    color="black",
                    fontsize="small",
                    fontweight="bold",
                )
            ax1.set_xlim(0, 1)
            ax1.set_xlabel("Estimated label posterior")
            ax1.set_ylabel("Example counts in bin")
            ax1.legend()
            plt.show()

    def get_full_range_bin_precision(
        self, c_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conf, preds = torch.max(self.sms, dim=1)
        class_idx = torch.where(preds == c_idx)[0]
        acc = preds[class_idx] == self.targets[class_idx]

        bin_precision = torch.zeros(self.n_bins)
        prop_in_bin = torch.zeros(self.n_bins)
        count_in_bin = torch.zeros(self.n_bins)
        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = (conf[class_idx] >= bin_lower) * (conf[class_idx] <= bin_upper)
            prop_in_bin[i] = in_bin.float().mean()
            count_in_bin[i] = in_bin.sum()
            if prop_in_bin[i] > 0:
                bin_precision[i] = acc[in_bin].float().mean()
        return bin_precision, count_in_bin
