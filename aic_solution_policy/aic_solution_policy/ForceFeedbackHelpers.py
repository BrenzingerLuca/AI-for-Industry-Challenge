import csv
from datetime import datetime, timezone
from pathlib import Path


class ForceFeedbackHelper:
	def __init__(self, logger):
		self._logger = logger
		self._last_wrench = None
		self._num_wrench_msgs = 0
		self._force_history = []
		self._force_stream_active = False
		self._force_csv_file = None
		self._force_csv_writer = None
		self._force_csv_path = None

	def on_wrench(self, msg):
		"""Silently store the latest wrench message."""
		self._last_wrench = msg
		self._num_wrench_msgs += 1

	def get_last_wrench(self):
		return self._last_wrench

	def get_num_wrench_msgs(self):
		return self._num_wrench_msgs

	def set_stream_active(self, active):
		self._force_stream_active = active

	def stream_tick(self, time_window=0.1):
		"""Collect one force sample and write to CSV while streaming is active."""
		if not self._force_stream_active:
			return None

		delta_forces, forces_gradient, abs_forces = self.get_force_feedback(time_window=time_window)
		self._log_force_feedback_csv(delta_forces, forces_gradient, abs_forces)
		return delta_forces, forces_gradient, abs_forces

	def start_csv_logging(self, data_dir):
		"""Create a per-run CSV file for force feedback samples."""
		data_dir = Path(data_dir)
		data_dir.mkdir(parents=True, exist_ok=True)

		file_name = f"force_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
		self._force_csv_path = data_dir / file_name
		self._force_csv_file = open(self._force_csv_path, "w", newline="", encoding="utf-8")
		self._force_csv_writer = csv.writer(self._force_csv_file)
		self._force_csv_writer.writerow([
			"wall_time_iso",
			"wall_time_s",
			"wrench_stamp_s",
			"abs_force_x",
			"abs_force_y",
			"abs_force_z",
			"abs_torque_x",
			"abs_torque_y",
			"abs_torque_z",
			"delta_force_x",
			"delta_force_y",
			"delta_force_z",
			"delta_torque_x",
			"delta_torque_y",
			"delta_torque_z",
			"grad_force_x",
			"grad_force_y",
			"grad_force_z",
			"grad_torque_x",
			"grad_torque_y",
			"grad_torque_z",
		])
		self._force_csv_file.flush()
		self._logger.info(f"Force CSV logging started: {self._force_csv_path}")

	def stop_csv_logging(self):
		"""Close the current force feedback CSV file, if open."""
		if self._force_csv_file is not None:
			self._force_csv_file.close()
			self._force_csv_file = None
			self._force_csv_writer = None
			if self._force_csv_path is not None:
				self._logger.info(f"Force CSV logging stopped: {self._force_csv_path}")
				self._force_csv_path = None

	def get_force_feedback(self, time_window=0.1):
		"""Return delta/gradient and absolute force-torque values for the configured time window."""
		if self._last_wrench is None:
			zero = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
			return zero, zero, zero

		abs_forces = (
			(
				self._last_wrench.wrench.force.x,
				self._last_wrench.wrench.force.y,
				self._last_wrench.wrench.force.z,
			),
			(
				self._last_wrench.wrench.torque.x,
				self._last_wrench.wrench.torque.y,
				self._last_wrench.wrench.torque.z,
			),
		)

		stamp = self._last_wrench.header.stamp
		current_time_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
		self._force_history.append((current_time_s, abs_forces))

		cutoff_time_s = current_time_s - time_window
		self._force_history = [(t, f) for t, f in self._force_history if t >= cutoff_time_s]

		if len(self._force_history) > 1:
			earlier_forces = self._force_history[0][1]
			delta_forces = (
				(
					abs_forces[0][0] - earlier_forces[0][0],
					abs_forces[0][1] - earlier_forces[0][1],
					abs_forces[0][2] - earlier_forces[0][2],
				),
				(
					abs_forces[1][0] - earlier_forces[1][0],
					abs_forces[1][1] - earlier_forces[1][1],
					abs_forces[1][2] - earlier_forces[1][2],
				),
			)

			forces_gradient = (
				(
					delta_forces[0][0] / time_window,
					delta_forces[0][1] / time_window,
					delta_forces[0][2] / time_window,
				),
				(
					delta_forces[1][0] / time_window,
					delta_forces[1][1] / time_window,
					delta_forces[1][2] / time_window,
				),
			)
			return delta_forces, forces_gradient, abs_forces

		zero = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
		return zero, zero, abs_forces

	def _log_force_feedback_csv(self, delta_forces, forces_gradient, abs_forces):
		"""Append one force feedback sample to the active CSV file."""
		if self._force_csv_writer is None or self._force_csv_file is None:
			return

		wall_now = datetime.now(timezone.utc)
		wrench_stamp_s = ""
		if self._last_wrench is not None:
			stamp = self._last_wrench.header.stamp
			wrench_stamp_s = f"{float(stamp.sec) + float(stamp.nanosec) * 1e-9:.9f}"

		self._force_csv_writer.writerow([
			wall_now.isoformat(),
			f"{wall_now.timestamp():.9f}",
			wrench_stamp_s,
			f"{abs_forces[0][0]:.9f}",
			f"{abs_forces[0][1]:.9f}",
			f"{abs_forces[0][2]:.9f}",
			f"{abs_forces[1][0]:.9f}",
			f"{abs_forces[1][1]:.9f}",
			f"{abs_forces[1][2]:.9f}",
			f"{delta_forces[0][0]:.9f}",
			f"{delta_forces[0][1]:.9f}",
			f"{delta_forces[0][2]:.9f}",
			f"{delta_forces[1][0]:.9f}",
			f"{delta_forces[1][1]:.9f}",
			f"{delta_forces[1][2]:.9f}",
			f"{forces_gradient[0][0]:.9f}",
			f"{forces_gradient[0][1]:.9f}",
			f"{forces_gradient[0][2]:.9f}",
			f"{forces_gradient[1][0]:.9f}",
			f"{forces_gradient[1][1]:.9f}",
			f"{forces_gradient[1][2]:.9f}",
		])
		self._force_csv_file.flush()
