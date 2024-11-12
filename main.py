from Autocali import AutomaticCalibration
from runner import CalibrationRunner
from report.result_report import Cbr
from docx import Document


def main():
    auto_calib = AutomaticCalibration()
    calibration_runner = CalibrationRunner(auto_calib)
    calibration_runner.run()
    report = Cbr(auto_calib, calibration_runner)
    doc = Document()
    report.generate_doc(doc = doc)

if __name__ == "__main__":
    main()
