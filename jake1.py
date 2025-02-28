from pandasai.helpers.openai_info import get_openai_callback
import matplotlib
import matplotlib.pyplot as plt
from pandasai.responses.response_parser import ResponseParser
from pandasai.connectors import PandasConnector
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import os
from PIL import Image
import re

# Set backend before import pyplot (Do not show a new windows after plotting)
matplotlib.use("Agg", force=False)

# Page setting
st.set_page_config(layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Compare credentials against the ones stored in the secrets TOML file.
        if username == st.secrets["login"]["username"] and password == st.secrets["login"]["password"]:
            st.session_state['logged_in'] = True
            st.success("Logged in successfully!")
        else:
            st.error("Incorrect username or password")
    st.stop()

# Replace it with your OPENAI API KEY
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# Init pandasai
llm = OpenAI(api_token=OPENAI_API_KEY)

if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'prompt_called' not in st.session_state:
    st.session_state['prompt_called'] = False




# Handle response messages according to type: dataframe, plot or text


class MyStResponseParser(ResponseParser):
    def __init__(self, context):
        super().__init__(context)

    def parse(self, result):
        if result['type'] == "dataframe":
            st.dataframe(result['value'])

        elif result['type'] == "plot":
            # Retrieve the plotting code from the result
            plot_code = result["value"]

            # Debug: show the original plotting code
            st.write("Original plot code:")
            st.code(plot_code)

            # Remove any lines that include plt.savefig or plt.close
            cleaned_lines = []
            for line in plot_code.splitlines():
                if "plt.savefig" in line or "plt.close" in line:
                    st.write("Removing line:", line)
                else:
                    cleaned_lines.append(line)
            plot_code_clean = "\n".join(cleaned_lines).lstrip()

            # Debug: show the cleaned plotting code
            st.write("Cleaned plot code:")
            st.code(plot_code_clean)

            # Set up the local environment with the original DataFrame
            df = st.session_state.get("original_df")
            if df is None:
                st.error("Original DataFrame not found in session state.")
                return

            dfs = [df]
            local_env = {"df": df, "dfs": dfs, "pd": pd, "plt": plt}

            try:
                exec(plot_code_clean, {}, local_env)
                fig = plt.gcf()
                if fig.axes:
                    st.pyplot(fig)
                else:
                    st.warning("Plot executed, but no plot axes generated.")
            except Exception as e:
                st.error(f"Plotting execution failed: {e}")
                st.code(plot_code_clean)
            finally:
                plt.clf()

        else:
            st.write(result['value'])
        return



@st.cache_data
def load_data():
    dataset_file = "Hourly_Data_2022.csv"
    df = pd.read_csv(dataset_file, low_memory=False)
    return df


if 'original_df' not in st.session_state:
    st.session_state['original_df'] = load_data()
    
# Tip: Adding Description for data fields to make GPT understand more easily, using in case you don't want to use GPT's automatic understanding mechanism
field_descriptions = {
    "Generic ID": "Generic or unique identifier for the row/record. (Custom user-defined.)",
    "Date": "Date of the settlement or operating day.",
    "HE": "Hour Ending for the settlement interval (commonly used in MISO settlements).",
    "DART P&L ($)": "Profit and loss from Day-Ahead and Real-Time market transactions combined ""(sometimes referred to as DART transactions).",
    "DA P&L ($)": "Profit and loss from Day-Ahead market transactions.",
    "P&L Leakage or Gain ($)": "Difference or ‘leakage’ between Day-Ahead and Real-Time P&L. (Custom user-defined.)",
    "DA Tot En (MWh)": "Total energy in megawatt-hours scheduled in the Day-Ahead market.",
    "RT Meter (MWh)": "Real-time metered energy in megawatt-hours.",
    "DA LMP ($/MWh)": "Locational Marginal Price in the Day-Ahead market.",
    "RT  LMP ($/MWh)": "Locational Marginal Price in the Real-Time market.",
    "DA Tot Rev ($)": "Total revenue from Day-Ahead market transactions.",
    "DA En Rev ($)": "Revenue from Day-Ahead energy transactions.",
    "RT En Rev ($)": "Revenue from Real-Time energy transactions.",
    "RT AS Rev ($)": "Revenue from Real-Time Ancillary Services.",
    "NRGA ($)": "Non-Regulated Generation Adjustment (if applicable).",
    "PV MWP ($)": "Make-whole payments for Price Volatility in Real-Time.",
    "RSG MWP ($)": "Make-whole payments for Revenue Sufficiency Guarantee (RSG).",
    "RSG Penalty ($)": "Penalties related to Revenue Sufficiency Guarantee (RSG).",
    "AS Penalty ($)": "Penalties related to Ancillary Services.",
    "RT Admin ($)": "Administrative costs for Real-Time market transactions.",
    "RT Tot Rev ($)": "Total revenue from Real-Time market transactions.",
    "DART Rev ($)": "Revenue from combined Day-Ahead and Real-Time (DART) transactions.",
    "RT Startup Cost ($)": "Costs associated with starting up generation in the Real-Time market.",
    "RT En Cost ($)": "Costs associated with Real-Time energy transactions.",
    "RT AS Cost ($)": "Costs associated with Real-Time Ancillary Services.",
    "RT Tot Cost ($)": "Total costs from Real-Time market transactions.",
    "FFDF Flag": "Indicator for Forced Fuel Deviation Flag (MISO settlements).",
    "EEEF Flag": "Indicator for Emergency Energy Exchange Flag (MISO settlements).",
    "RSG MWP Elig Flag": "Eligibility flag for Revenue Sufficiency Guarantee make-whole payments.",
    "DA Map Flag": "Indicator for Day-Ahead market mapping in MISO settlements.",
    "RT ORSGP Flag": "Indicator for Real-Time Operating Reserve Shortfall Guarantee Payment.",
    "RT REG ($/MWh)": "Price per megawatt-hour for Real-Time Regulation Services.",
    "RT REG (MWh)": "Megawatt-hours of Real-Time Regulation Services (hourly).",
    "RT REG Cost ($)": "Costs associated with Real-Time Regulation Services. (Custom definition.)",
    "RT REG Rev ($)": "Revenue from Real-Time Regulation Services. (Custom definition.)",
    "RT SPIN ($/MWh)": "Price per megawatt-hour for Real-Time Spinning Reserve Services.",
    "RT SPIN (MWh)": "Megawatt-hours of Real-Time Spinning Reserve Services.",
    "RT SPIN Cost ($)": "Costs associated with Real-Time Spinning Reserve Services. (Custom definition.)",
    "RT SPIN Rev ($)": "Revenue from Real-Time Spinning Reserve Services. (Custom definition.)",
    "RT Supp ($/MWh)": "Price per megawatt-hour for Real-Time Supplemental Reserve Services.",
    "RT SUPP (MWh)": "Megawatt-hours of Real-Time Supplemental Reserve Services.",
    "RT SUPP Cost ($)": "Costs associated with Real-Time Supplemental Reserve Services. (Custom definition.)",
    "RT SUPP Rev ($)": "Revenue from Real-Time Supplemental Reserve Services. (Custom definition.)",
    "DA  Admin ($)": "Administrative costs for Day-Ahead market transactions.",
    "DA AS Cost ($)": "Costs associated with Day-Ahead Ancillary Services.",
    "DA AS Rev ($)": "Revenue from Day-Ahead Ancillary Services.",
    "DA Startup Cost ($)": "Costs associated with starting up generation in the Day-Ahead market.",
    "DA En Cost ($)": "Costs associated with Day-Ahead energy transactions.",
    "DA  Control Status": "Control status for Day-Ahead market participation (MISO scheduling).",
    "MR MWP Elig": "Eligibility indicator for Make-Whole Payments in the MISO market (generic).",
    "MR ELMP MWP ($)": "Make-whole payments for Extended LMP in the Market (Real-Time).",
    "DB ELMP MWP ($)": "Make-whole payments for Extended LMP in the Day-Ahead market.",
    "AO DA ELMP MWP ($)": "Make-whole payments for Extended LMP in the Day-Ahead market for Asset Owners.",
    "DA RSG Elg Flag": "Eligibility flag for Day-Ahead RSG make-whole payments.",
    "DA Mitigation Flag": "Indicator for Day-Ahead market mitigation (MISO).",
    "DA RSG MWP ($)": "Make-whole payments for Revenue Sufficiency Guarantee in the Day-Ahead market.",
    "DA Tot Cost ($)": "Total costs from Day-Ahead market transactions.",
    "DA GFA LS Credit ($)": "Day-Ahead GFA Load Serving Credit. (Grandfathered Agreement credit, custom definition.)",
    "DA GFA CG Credit ($)": "Day-Ahead GFA Congestion Credit. (Grandfathered Agreement credit, custom definition.)",
    "Total DA GFA Credit ($)": "Total Day-Ahead GFA credits (sum of LS, CG, etc.).",
    "DA P&L with GFA Credit ($)": "Day-Ahead profit/loss including GFA (Grandfathered Agreement) credits.",
    "DA MAP ($)": "Market Administration Price in the Day-Ahead market (MISO).",
    "RT ORSGP ($)": "Real-Time Operating Reserve Shortfall Guarantee Payment (dollar amount).",
    "RT NXE ($)": "Real-Time Non-Exempt Energy charges.",
    "RT EXE ($)": "Real-Time Exempt Energy charges.",
    "RT NXE (MWh)": "Megawatt-hours of Real-Time Non-Exempt Energy.",
    "RT EXE (MWh)": "Megawatt-hours of Real-Time Exempt Energy.",
    "RT DFE (MWh)": "Megawatt-hours of Real-Time Demand Forecast Error.",
    "RT EXE/DFE Pen ($)": "Penalties for Real-Time Exempt Energy and Demand Forecast Error.",
    "RT CRDF Pen ($)": "Penalties for Real-Time Capacity Resource Deficiency.",
    "Tot DART Reg Rev ($)":"Total Day-Ahead and Real-Time Regulation revenue (summed together).",
    "DA Reg ($)": "Revenue from Day-Ahead Regulation Services.",
    "RT Reg ($)": "Revenue from Real-Time Regulation Services.",
    "Reg Penalty ($)": "Penalties related to Regulation Services. (Custom definition.)",
    "Reg Mil Penalty ($)": "Penalties related to Regulation Mileage.",
    "FMPTF": "Financial Market Participation Transaction Fee (MISO).",
    "RT Hrly Reg (MWh)": "Megawatt-hours of Real-Time Hourly Regulation Services.",
    "RT Net Reg (MWh)": "Net megawatt-hours of Real-Time Regulation Services (e.g., net of instructions).",
    "MISO Reg Pen Rate ($/MWh)": "Penalty rate for Regulation Services in MISO (if applicable).",
    "5M Reg (MWh)": "Megawatt-hours of 5-Minute Regulation Services.",
    "Add Reg Mil Vol (MWh)": "Additional Regulation Mileage Volume.",
    "RT Reg Mil MCP ($/MWh)": "Price per megawatt-hour for Real-Time Regulation Mileage.",
    "5M Desired Mileage (MWh)": "Desired mileage for 5-Minute Regulation Services.",
    "5M Actual Mileage (MWh)": "Actual mileage for 5-Minute Regulation Services.",
    "Reg Perf (%)": "Performance percentage for Regulation Services (comparing actual vs. instructed).",
    "5M Reg Follow": "Indicator for whether 5-Minute Regulation is following dispatch signals.",
    "5M FMPTF": "Financial Market Participation Transaction Fee for 5-Minute Services. (Custom.)",
    "5m Cmode": "Control mode for 5-Minute Regulation Services. (Custom definition.)",
    "Av Setpoint (MW)": "Average setpoint in megawatts (real-time dispatch).",
    "5M Meter (MWh)": "5-minute metered energy in megawatt-hours.",
    "5M Adj Meter (MWh)": "Adjusted 5-minute metered energy in megawatt-hours.",
    "EXE Threshold (MWh)": "Threshold for Exempt Energy in megawatt-hours.",
    "DFE Threshold (MWh)": "Threshold for Demand Forecast Error in megawatt-hours.",
    "CB Exempt": "Indicator for capacity-based exemption (custom placeholder).",
    "Undeploy Reg Mil Amt ($)": "Amount for undeployed regulation mileage (if not fully deployed).",
    "Tot Mil MWP ($)": "Total make-whole payments for regulation mileage (Day-Ahead + Real-Time).",
    "DA Mil MWP ($)": "Day-Ahead make-whole payments for regulation mileage.",
    "RT Mil MWP ($)": "Real-Time make-whole payments for regulation mileage.",
    "5M Mileage Reg MCP ($/MWh)": "Price per MWh for 5-minute Regulation Mileage (MISO).",
    "DART P&L with GFA Credit ($)": "DART (Day-Ahead and Real-Time) profit/loss with GFA credits included. (Custom.)",
    "Total DART GFA Credit ($)": "Total GFA credits (Day-Ahead + Real-Time) for DART. (Custom.)",
    "DA GFA CG Credit ($)2": "Day-Ahead GFA Congestion Credit #2 (custom or user-defined).",
    "DA GFA LS Credit ($)3": "Day-Ahead GFA Load Serving Credit #3 (custom or user-defined).",
    "RT GFA CG Credit ($)": "Real-Time GFA Congestion Credit (custom or user-defined).",
    "RT GFA LS Credit ($)": "Real-Time GFA Load Serving Credit (custom or user-defined).",
    "DA GFA Buy Vol (MWh)": "Day-Ahead GFA Buy Volume in MWh (custom or user-defined).",
    "DA GFA Sell Vol (MWh)": "Day-Ahead GFA Sell Volume in MWh (custom or user-defined).",
    "RT GFA Buy Vol (MWh)": "Real-Time GFA Buy Volume in MWh (custom or user-defined).",
    "RT GFA Sell Vol (MWh)": "Real-Time GFA Sell Volume in MWh (custom or user-defined).",
    "RT Ramp Rev ($)": "Revenue from Real-Time Ramp Capability Services.",
    "RT UpRamp (MWh)": "Megawatt-hours of Real-Time Up Ramp Capability.",
    "RT DnRamp (MWh)": "Megawatt-hours of Real-Time Down Ramp Capability.",
    "RT UpRamp ($/MWh)": "Price per megawatt-hour for Real-Time Up Ramp Capability.",
    "RT DnRamp ($/MWh)": "Price per megawatt-hour for Real-Time Down Ramp Capability.",
    "DA Ramp Rev ($)": "Revenue from Day-Ahead Ramp Capability Services.",
    "DA UpRamp (MWh)": "Megawatt-hours of Day-Ahead Up Ramp Capability.",
    "DA DnRamp (MWh)": "Megawatt-hours of Day-Ahead Down Ramp Capability.",
    "DA UpRamp ($/MWh)": "Price per megawatt-hour for Day-Ahead Up Ramp Capability.",
    "DA DnRamp ($/MWh)": "Price per megawatt-hour for Day-Ahead Down Ramp Capability.",}

def main_page():
    st.header("MISO AI Analytics Tool")

    # Button to load the dataset into memory
    if st.button("Load Dataset into Memory", type="primary"):
        st.session_state['data_loaded'] = True

    if st.session_state.get('data_loaded', False):
        # Display a sample of the original DataFrame
        with st.expander("Check Data Reading Log", expanded=False):
            st.write(st.session_state['original_df'].head(3))
            st.markdown(st.session_state['original_df'].info())

        st.write("### Ask me anything about your Data:")

        # Input for user prompt
        prompt = st.text_input("Enter your prompt", value="What is DART P&L ($) average in January 2022")

        # Reset the DataFrame to the original before each query
        df = st.session_state['original_df'].copy()

        # Initialize the connector and agent with the reset DataFrame
        connector = PandasConnector({'original_df': df})
        agent = SmartDataframe(connector, config={
            "llm": llm,
            "save_charts": False,       # CRITICAL: Prevent local saving
            "open_charts": False,       # Avoid external plot rendering
            "custom_whitelisted_dependencies": ["matplotlib"],
            "conversational": False,
            "response_parser": MyStResponseParser,
            "enforce_privacy": True,
        })


        # Button to send the prompt
        if st.button("Send Message"):
            if prompt:
                st.write("👻 Response:")
                with st.spinner("Generating response..."):
                    with get_openai_callback() as call_back_info:
                        chat_response = agent.chat(prompt)
                        st.write("📚 What happened behind:")
                        st.code(agent.last_code_executed)
                        st.write(call_back_info)
            else:
                st.warning("Please enter a prompt")

if __name__ == '__main__':
    main_page()
