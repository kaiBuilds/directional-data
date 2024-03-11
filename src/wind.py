import re
import numpy as np


class WindCleaner:
    def __init__(
        self,
        raw_data: str,
        cleaned_data: str,
    ):
        self.raw_data: str = raw_data
        self.result_file: str = cleaned_data
        self.useful_fields = [
            "DTG",
            "LOCATION",
            "LATITUDE",
            "LONGITUDE",
            "ALTITUDE",
            "FF_SENSOR_10",
            "DD_10",
            "DD_STD_10",
        ]

        self.pattern = re.compile(
            r"(?P<DTG>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
            r"(?P<LOCATION>\S+)\s+"
            r"(?P<NAME>.*?)\s{2,}"
            r"(?P<LATITUDE>-?\d+\.\d+)\s+"
            r"(?P<LONGITUDE>-?\d+\.\d+)\s+"
            r"(?P<ALTITUDE>-?\d+\.\d+)\s+"
            r"(?P<FF_SENSOR_10>\d+\.\d+)?\s+"
            r"(?P<FF_10M_10>\d+\.\d+)?\s+"
            r"(?P<DD_10>\d+\.\d+)?\s+"
            r"(?:\s*(?P<DDN_10>\d+\.\d+))?\s*"
            r"(?:\s*(?P<DD_STD_10>\d+\.\d+))?\s*"
            r"(?:\s*(?P<DDX_10>\d+\.\d+))?\s*"
            r"(?:\s*(?P<FF_10M_STD_10>\d+\.\d+))?\s*"
            r"(?:\s*(?P<FX_10M_10>\d+\.\d+))?\s+"
            r"(?:\s*(?P<FX_10M_MD_10>\d+\.\d+))?\s*"
            r"(?:\s*(?P<FX_SENSOR_10>\d+\.\d+))?\s*"
            r"(?:\s*(?P<FX_SENSOR_MD_10>\d+\.\d+))?\s*"
            r"(?P<SQUALL_10>\d)?\s*"
        )

    def run(self):
        with open(self.raw_data, "r") as raw:
            with open(self.result_file, "w") as clean:
                clean.write(",".join(self.useful_fields) + "\n")

                for i, row in enumerate(raw):
                    if i < 24:
                        continue
                    if i % 1000 == 0:
                        print(f"Processing line {i}")
                    processed = self.process_data(row)
                    if processed:
                        clean.write(",".join(processed) + "\n")

    def process_data(self, row: list) -> list:
        """Match the pattern and extract the useful fields."""
        match = self.pattern.search(row)
        if match:
            matches = match.groupdict()
            if matches["DD_10"]:
                matches["DD_10"] = str(np.deg2rad(float(matches["DD_10"])))
            if matches["DD_STD_10"]:
                matches["DD_STD_10"] = str(np.deg2rad(float(matches["DD_STD_10"])))
            return [matches[key] if matches[key] else "" for key in self.useful_fields]
        else:
            return []
            # raise ValueError(f"Pattern not found in {row}")


if __name__ == "__main__":
    processor = WindCleaner(
        raw_data="/workspaces/directional/data/kis_tow/kis_tow_202402",
        cleaned_data="/workspaces/directional/data/kis_tow/clean_kis_tow_202402.csv",
    )
    processor.run()
