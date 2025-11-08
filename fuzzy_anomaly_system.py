
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyAnomalyDetector:
    """
    Fuzzy Inference System (FIS) for anomaly detection in multivariate time series.

    Inputs:
        - forecast_error: normalized forecast error in [0, 1]
        - variance_change: normalized variance change in [0, 1]
        - correlation_change: normalized correlation change in [0, 1]

    Output:
        - anomaly_level: anomaly level in [0, 10]
    """

    def __init__(self):
        self.forecast_error = ctrl.Antecedent(np.linspace(0, 1, 101), "forecast_error")
        self.variance_change = ctrl.Antecedent(np.linspace(0, 1, 101), "variance_change")
        self.correlation_change = ctrl.Antecedent(np.linspace(0, 1, 101), "correlation_change")

        self.anomaly_level = ctrl.Consequent(np.linspace(0, 10, 101), "anomaly_level")

        self._define_membership_functions()
        self._define_rules()
        self._build_control_system()

    def _define_membership_functions(self):
        """
        Define membership functions for all linguistic variables.
        """

        for var in [self.forecast_error, self.variance_change, self.correlation_change]:
            var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 0.4])
            var["medium"] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
            var["high"] = fuzz.trimf(var.universe, [0.6, 1.0, 1.0])

        self.anomaly_level["normal"] = fuzz.trimf(self.anomaly_level.universe, [0, 0, 3])
        self.anomaly_level["slightly_anomalous"] = fuzz.trimf(
            self.anomaly_level.universe, [1, 3, 5]
        )
        self.anomaly_level["moderately_anomalous"] = fuzz.trimf(
            self.anomaly_level.universe, [3, 5, 7]
        )
        self.anomaly_level["strongly_anomalous"] = fuzz.trimf(
            self.anomaly_level.universe, [6, 10, 10]
        )

    def _define_rules(self):
        """
        Define the fuzzy rule base (Mamdani).
        """

        fe = self.forecast_error
        vc = self.variance_change
        cc = self.correlation_change
        al = self.anomaly_level

        rules = []

        rules.append(
            ctrl.Rule(
                fe["low"] & vc["low"] & cc["low"],
                al["normal"],
                label="R1_normal_all_low",
            )
        )

        rules.append(
            ctrl.Rule(
                fe["medium"] & vc["low"] & cc["low"],
                al["slightly_anomalous"],
                label="R2_slightly_anom_medium_fe",
            )
        )
        rules.append(
            ctrl.Rule(
                fe["low"] & vc["medium"] & cc["low"],
                al["slightly_anomalous"],
                label="R3_slightly_anom_medium_vc",
            )
        )
        rules.append(
            ctrl.Rule(
                fe["low"] & vc["low"] & cc["medium"],
                al["slightly_anomalous"],
                label="R4_slightly_anom_medium_cc",
            )
        )

        rules.append(
            ctrl.Rule(
                fe["medium"] & vc["medium"],
                al["moderately_anomalous"],
                label="R5_moderate_fe_vc",
            )
        )
        rules.append(
            ctrl.Rule(
                fe["medium"] & cc["medium"],
                al["moderately_anomalous"],
                label="R6_moderate_fe_cc",
            )
        )
        rules.append(
            ctrl.Rule(
                vc["medium"] & cc["medium"],
                al["moderately_anomalous"],
                label="R7_moderate_vc_cc",
            )
        )

        rules.append(
            ctrl.Rule(
                fe["high"] & vc["high"],
                al["strongly_anomalous"],
                label="R8_strong_fe_vc",
            )
        )
        rules.append(
            ctrl.Rule(
                fe["high"] & cc["high"],
                al["strongly_anomalous"],
                label="R9_strong_fe_cc",
            )
        )
        rules.append(
            ctrl.Rule(
                vc["high"] & cc["high"],
                al["strongly_anomalous"],
                label="R10_strong_vc_cc",
            )
        )

        rules.append(
            ctrl.Rule(
                fe["high"] & vc["medium"],
                al["strongly_anomalous"],
                label="R11_high_fe_medium_vc",
            )
        )
        rules.append(
            ctrl.Rule(
                fe["medium"] & vc["high"],
                al["strongly_anomalous"],
                label="R12_medium_fe_high_vc",
            )
        )
        rules.append(
            ctrl.Rule(
                fe["high"] & cc["medium"],
                al["strongly_anomalous"],
                label="R13_high_fe_medium_cc",
            )
        )
        rules.append(
            ctrl.Rule(
                vc["high"] & cc["medium"],
                al["strongly_anomalous"],
                label="R14_high_vc_medium_cc",
            )
        )

        self.rules = rules

    def _build_control_system(self):
        """
        Build the fuzzy control system and simulator.
        """
        system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(system)

    def evaluate(self, forecast_error, variance_change, correlation_change):
        """
        Evaluate the FIS for a given set of normalized indicators.

        Parameters:
            forecast_error (float): normalized forecast error [0, 1]
            variance_change (float): normalized variance change [0, 1]
            correlation_change (float): normalized correlation change [0, 1]

        Returns:
            crisp_value (float): defuzzified anomaly level (0 to 10)
            label (str): dominant linguistic label
        """
        fe = float(np.clip(forecast_error, 0.0, 1.0))
        vc = float(np.clip(variance_change, 0.0, 1.0))
        cc = float(np.clip(correlation_change, 0.0, 1.0))

        self.simulator.input["forecast_error"] = fe
        self.simulator.input["variance_change"] = vc
        self.simulator.input["correlation_change"] = cc

        self.simulator.compute()

        crisp_value = float(self.simulator.output["anomaly_level"])

        membership_values = {
            name: fuzz.interp_membership(
                self.anomaly_level.universe, self.anomaly_level[name].mf, crisp_value
            )
            for name in self.anomaly_level.terms
        }

        label = max(membership_values, key=membership_values.get)

        return crisp_value, label
