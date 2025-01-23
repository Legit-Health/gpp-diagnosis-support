import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def assess_impact_hypotheses(df, alpha=0.05, exact_thresh=25):
    """Analysis of the impact of the device on practitioners' performance"""
    total = df.shape[0]

    # Get values that are correct without and with device
    answers = []
    query_base = '`Is correct before device`=="{}" and `Is correct after device`=="{}"'
    query_args = [
        ["correct", "correct"],
        ["correct", "incorrect"],
        ["incorrect", "correct"],
        ["incorrect", "incorrect"],
    ]
    for args in query_args:
        answers.append(df.query(query_base.format(*args)).shape[0])

    # Build the matrix
    data = np.array([answers[:2], answers[2:]])

    df = pd.DataFrame(
        data,
        columns=["Answer with device (correct)", "Answer with device (incorrect)"],
        index=["Answer wihout device (correct)", "Answer wihout device (incorrect)"],
    )

    # McNemar's Test
    # If the diagonal is equal or less than 25
    # then apply the Exact McNemar test (Binomial distribution)
    if np.diag(data).sum() < exact_thresh:
        result = mcnemar(data, exact=True, correction=False)
    else:
        result = mcnemar(data, exact=False, correction=False)
    p_value = result.pvalue
    print("\nP-Value:", p_value)

    h0 = "the device has an impact on the improvement of diagnostic capabilities"
    h1 = "the has an impact on the improvement of diagnostic capabilities"
    print(f"H0 <- There is no significant evidence to claim that {h0}")
    print(f"H1 <- There is significant evidence to claim that {h1}")

    if p_value < float(alpha):
        # If the p-value is less than the significance level
        # then reject the null hypothesis
        action = "reject"
        print(
            f"\nSince the p-value = {p_value:.6f} < α ({alpha:.0%}),"
            f" we {action} the null hypothesis"
            f" and claim that we have 95% confidence that {h0}"
        )
    else:
        # If the p-value is greater than the significance level
        # then accept the alternative hypothesis
        action = "cannot reject"
        print(
            f"\nSince the p-value = {p_value:.6f} > α ({alpha:.0%}),"
            f" we {action} the null hypothesis"
            f" as there is no significant evidence to claim that {h1}"
        )

    # Percentage of cases that are correct without and with device
    reinforcement_rate = data[0, 0] / total
    print(f"- Reinforcement of practitioners' performance: {reinforcement_rate:.2%}")

    # Percentage of cases that are incorrect without device and correct with device
    correction_rate = data[1, 0] / total
    print(f"- Improvement of practitioners' performance: {correction_rate:.2%}")

    # Percentage of cases that are incorrect without device and with device
    double_failure_rate = data[1, 1] / total
    print(f"- Unaffected performance: {double_failure_rate:.2%}")

    # Percentage of cases that are correct without device and correct with device
    failure_rate = data[0, 1] / total
    print(f"- Negative impact on performance: {failure_rate:.2%}")


def assess_impact_on_pathology(df):
    df_before = df[["Correct condition", "Is correct before device"]]
    table1 = (
        df_before.value_counts()
        .to_frame("Count_before_device")
        .reset_index()
        .assign(
            Total_count_before_device=lambda row: row.groupby("Correct condition")[
                "Count_before_device"
            ].transform("sum"),
            Rate_before_device=lambda row: (
                (row["Count_before_device"] / row["Total_count_before_device"]) * 100
            ).round(2),
        )
    )
    table1.sort_values(
        by=["Correct condition", "Is correct before device"],
        ascending=False,
        inplace=True,
    )
    table1.reset_index(drop=True, inplace=True)
    # print(table1)

    df_after = df[["Correct condition", "Is correct after device"]]
    table2 = (
        df_after.value_counts()
        .to_frame("Count_after_device")
        .reset_index()
        .assign(
            Total_count_after_device=lambda row: row.groupby("Correct condition")[
                "Count_after_device"
            ].transform("sum"),
            Rate_after_device=lambda row: (
                (row["Count_after_device"] / row["Total_count_after_device"]) * 100
            ).round(2),
        )
        .sort_values(
            by=["Correct condition", "Is correct after device"], ascending=False
        )
        .reset_index(drop=True)
    )
    # print(table2)

    # Combining before and after
    result = pd.concat([table1, table2.iloc[:, 1:]], axis=1)
    print(result)

    return result


if __name__ == "__main__":

    # Loading the data
    df_result = pd.read_csv("data/BI_results_data.csv")

    # Number of images
    n_images = df_result["Image"].nunique()
    n_cases = df_result.shape[0]
    print(f"Number of unique images in the dataset: {n_images}\n")
    print(f"Total cases analyzed in the dataset: {n_cases}\n")

    # Number of practitioners per specialty
    print("Number of practitioners per specialty:\n")
    print(
        df_result[["Fullname", "Specialty"]]
        .drop_duplicates()["Specialty"]
        .value_counts()
        .to_frame("Nº of practioners")
        .reset_index()
    )

    # Number of images analyzed per  practitioner
    print("\n\nNumber of images analyzed per practitioner\n")
    print(
        df_result.groupby(["Fullname"])["Image"]
        .nunique()
        .to_frame("Nº of images analyzed per pracitioner")
        .reset_index()
        .sort_values(by="Nº of images analyzed per pracitioner", ascending=True)
        .reset_index(drop=True)
    )
    # Number of images analyzed per specialty
    # Both images are analyzed by both specialties
    print("\nNumber of images analyzed per specialty")
    print(
        df_result[["Specialty", "Image"]]
        .drop_duplicates()["Specialty"]
        .value_counts()
        .to_frame("Nº images analyzed per specialty")
        .reset_index()
    )

    # Get overall % of accuracy
    for stage in ["before", "after"]:
        col_name = f"Is correct {stage} device"
        new_col_name = f"Record count {stage} device"
        print(
            df_result[col_name]
            .value_counts()
            .to_frame(new_col_name)
            .reset_index()
            .assign(Rate=lambda x: x[new_col_name] / x[new_col_name].sum() * 100)
            .round(2)
        )

    ######################################################################
    # Analysis of the impact of the device on practitioners' performance #
    ######################################################################

    assess_impact_hypotheses(df_result)

    # Analysis per pathology
    for pathology in df_result["Correct condition"].unique():
        # Get total rows (values)
        df_result_pathology = df_result.query("`Correct condition`==@pathology")
        assess_impact_hypotheses(df_result_pathology)

    # Analysis per specialty
    specialty_dict = {"Medicina general": "PCP", "Dermatología": "Dermatologist"}
    specialty_list = df_result["Specialty"].unique()
    for specialty in specialty_list:
        print(f"\nAssessing performance of: {specialty}\n")

        df_result_specialty = df_result.query("Specialty==@specialty")

        # Get % of accuracy by specialty
        print(
            df_result_specialty["Is correct before device"]
            .value_counts()
            .to_frame("Record count before device")
            .reset_index()
            .assign(
                Rate=lambda x: x["Record count before device"]
                / x["Record count before device"].sum()
                * 100
            )
            .round(2)
        )
        print(
            df_result_specialty["Is correct after device"]
            .value_counts()
            .to_frame("Record count after device")
            .reset_index()
            .assign(
                Rate=lambda x: x["Record count after device"]
                / x["Record count after device"].sum()
                * 100
            )
            .round(2)
        )

        # P-values
        assess_impact_hypotheses(df_result_specialty)

        # Uncomment this part if you want to get the p-values per pathology
        # for pathology in df_result_specialty["Correct condition"].unique():
        #     # Get total rows (values)
        #     df_result_pathology = df_result_specialty.query(
        #         "`Correct condition`==@pathology"
        #     )
        #     assess_impact_hypotheses(df_result_pathology)

    #########################
    # Exporting the results #
    #########################

    export_cols = ["Correct condition", "Rate_before_device", "Rate_after_device"]

    # All pathologies and specialties
    hcp_table = assess_impact_on_pathology(df_result)
    hcp_table = hcp_table.query("`Is correct after device`=='correct'")
    hcp_table.sort_values(by="Correct condition", inplace=True)
    hcp_table[export_cols].to_csv("results/HCP_performance.csv", index=False)

    # Per specialty
    for specialty in specialty_list:
        print("\nAssessing performance impact per pathology: ", specialty)
        df_result_specialty = df_result.query("Specialty==@specialty")
        specialty_table = assess_impact_on_pathology(df_result_specialty)
        specialty_table = specialty_table.query("`Is correct after device`=='correct'")
        specialty_table.sort_values(by="Correct condition", inplace=True)
        table_name = f"{specialty_dict[specialty]}_performance.csv"
        specialty_table[export_cols].to_csv(f"results/{table_name}", index=False)
