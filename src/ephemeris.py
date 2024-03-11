import csv
import numpy as np
from datetime import datetime
from tqdm import tqdm


class EphemerisCleaner:
    def __init__(
        self,
        raw_data: str,
        cleaned_data: str,
    ):
        self.raw_data: str = raw_data
        self.result_file: str = cleaned_data
        self.date = datetime.today()

        # Ephemeris in cartesian coordinates
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.dx: float = 0.0
        self.dy: float = 0.0
        self.dz: float = 0.0

        # Ephemeris in spherical coordinates
        self.altitude: float = 0.0
        self.theta: float = 0.0
        self.phi: float = 0.0
        self.daltitude: float = 0.0
        self.dtheta: float = 0.0
        self.dphi: float = 0.0

    def run(self):
        with open(self.raw_data, "r") as raw:
            with open(self.result_file, "w") as clean:
                reader = csv.reader(raw, delimiter=" ", skipinitialspace=True)
                writer = csv.writer(clean, delimiter=",")
                writer.writerow(
                    [
                        "Date",
                        "Altitude",
                        "Theta",
                        "Phi",
                        "dAltitude",
                        "dTheta",
                        "dPhi",
                    ]
                )
                for i, row in enumerate(tqdm(reader)):
                    if i < 4 or i % 4 != 0:  # Skip the first 4 rows and every 4th row
                        continue
                    processed = self.process_data(row)
                    writer.writerow(processed)

    def process_data(self, row: list) -> list:
        """Process the data from the raw file into spherical coordinates"""
        self.date = datetime.strptime(row[0].split(".")[0], "%Y%j%H%M%S")
        self.x, self.y, self.z = [float(r) for r in row[1:4]]
        self.dx, self.dy, self.dz = [float(r) for r in row[4:7]]
        self.altitude = (self.x**2 + self.y**2 + self.z**2) ** 0.5
        self.theta = np.arctan(self.y / self.x)
        self.phi = np.arccos(self.z / self.altitude) * np.sign(self.x)
        x = self.solve()

        self.daltitude = x[0]
        self.dtheta = x[1]
        self.dphi = x[2]
        return [
            self.date,
            self.altitude,
            self.theta,
            self.phi,
            self.daltitude,
            self.dtheta,
            self.dphi,
        ]

    def solve(self) -> np.ndarray:
        """Solve the system of linear equations to get the derivatives of the spherical coordinates."""
        b = np.array([self.dx, self.dy, self.dz]).reshape(3, 1)
        A = np.array(
            [
                [
                    np.cos(self.theta) * np.sin(self.phi),
                    self.altitude * np.sin(self.theta) * np.sin(self.phi),
                    self.altitude * np.cos(self.theta) * np.cos(self.phi),
                ],
                [
                    np.sin(self.theta) * np.sin(self.phi),
                    self.altitude * np.cos(self.theta) * np.sin(self.phi),
                    self.altitude * np.sin(self.theta) * np.cos(self.phi),
                ],
                [np.cos(self.phi), 0.0, -self.altitude * np.sin(self.phi)],
            ]
        ).reshape(3, 3)

        return np.linalg.solve(A, b).reshape(3)


if __name__ == "__main__":
    processor = EphemerisCleaner(
        raw_data="/workspaces/directional/extract",
        cleaned_data="ephemeris.csv",
    )
    processor.run()
