import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset with new cache method
@st.cache_data
def load_data():
    data = pd.read_csv('finance_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data.copy()  # Return a copy to avoid mutating the cached object

data = load_data()

# Title and description
st.title("22AIA-WEBSPIRITS")
st.title("Personal Finance Dashboard")
st.write("An interactive dashboard to track and manage your personal finances.")

# Display raw data
st.write("### Raw Data")
st.dataframe(data)

# Summary statistics
st.write("### Summary")
total_income = data[data['Type'] == 'Income']['Amount'].sum()
total_expense = data[data['Type'] == 'Expense']['Amount'].sum()
net_savings = total_income - total_expense

st.metric("Total Income", f"${total_income:,.2f}")
st.metric("Total Expense", f"${total_expense:,.2f}")
st.metric("Net Savings", f"${net_savings:,.2f}")

# Monthly analysis
st.write("### Monthly Analysis")
data['Month'] = data['Date'].dt.to_period('M')
monthly_summary = data.groupby(['Month', 'Type'])['Amount'].sum().unstack().fillna(0)
monthly_summary['Net Savings'] = monthly_summary['Income'] - monthly_summary['Expense']

# Plot monthly income vs expense
st.write("#### Income vs Expense by Month")
fig, ax = plt.subplots()
monthly_summary[['Income', 'Expense']].plot(kind='bar', ax=ax)
ax.set_ylabel("Amount ($)")
st.pyplot(fig)

# Pie chart of expenses by category
st.write("### Expense Breakdown by Category")
expense_by_category = data[data['Type'] == 'Expense'].groupby('Category')['Amount'].sum()
fig, ax = plt.subplots()
expense_by_category.plot(kind='pie', autopct='%1.1f%%', ax=ax)
ax.set_ylabel("")
st.pyplot(fig)

# Add new entry
st.write("### Add New Entry")
with st.form("entry_form"):
    date = st.date_input("Date")
    entry_type = st.selectbox("Type", ["Income", "Expense"])
    category = st.text_input("Category")
    amount = st.number_input("Amount", min_value=0.0, format="%0.2f")
    submit = st.form_submit_button("Add Entry")
    
    if submit:
        new_entry = pd.DataFrame([[date, entry_type, category, amount]], columns=['Date', 'Type', 'Category', 'Amount'])
        data = data.append(new_entry, ignore_index=True)
        data.to_csv('finance_data.csv', index=False)
        st.success("Entry added successfully!")
        st.experimental_rerun()

# Display updated data
st.write("### Updated Data")
st.dataframe(data)
