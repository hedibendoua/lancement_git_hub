# Show different content based on the user's email address.
if st.experimental_user.email == "jane@examples.com":
    display_jane_content()
elif st.experimental_user.email == "adam@example.com":
    display_adam_content()
else:
    st.write("Please contact us to get access!")

# Get dictionaries of cookies and headers
st.context.cookies
st.context.headers