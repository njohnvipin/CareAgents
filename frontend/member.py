
import requests
import streamlit as st

# Existing FastAPI backend.
API_URL = "http://127.0.0.1:8000"

# Synthetic member used for local development.
MEMBER_ID = "SARAH-1001"


def show_member_portal():
    st.title("💬 Member Portal")
    st.caption("Your AI-powered healthcare assistant")

    # Initialize chat history.
    if "member_messages" not in st.session_state:
        st.session_state.member_messages = []

    # Enter a development JWT without hardcoding it.
    token = st.text_input(
        "Member JWT",
        key="member_jwt",
        type="password",
        help="Paste your locally generated member JWT.",
    )

    st.caption(f"Test member: {MEMBER_ID}")

    # Quick questions.
    st.subheader("Quick Questions")

    questions = {
        "📋 Claim Status": "What is the status of my MRI claim CLM-8892?",
        "🏥 Coverage": "What does my health plan cover?",
        "💳 Billing": "Can you explain my medical bill?",
        "📝 Authorization": "Does an MRI require prior authorization?",
    }

    columns = st.columns(4)
    selected_question = None

    for column, (label, sample_question) in zip(
        columns, questions.items()
    ):
        with column:
            if st.button(
                label,
                key=f"quick_{label}",
                use_container_width=True,
            ):
                selected_question = sample_question

    st.divider()
    st.subheader("Ask CareAgents")

    # Display previous chat messages.
    for message in st.session_state.member_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message.get("pending_review"):
                st.warning("This case requires human review.")

            if message.get("case_id"):
                st.caption(f"Case ID: {message['case_id']}")

    # Accept a question from the chat box.
    typed_question = st.chat_input(
        "Ask about your claims, coverage, or benefits..."
    )

    question = typed_question or selected_question

    if question:
        if not token:
            st.warning("Enter your Member JWT before chatting.")
        else:
            # Display and save the member's question.
            st.session_state.member_messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.write(question)

            # Send the question to FastAPI.
            with st.chat_message("assistant"):
                with st.spinner("CareAgents is processing..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/v1/chat",
                            headers={
                                "Authorization": f"Bearer {token}",
                            },
                            json={
                                "question": question,
                                "member_id": MEMBER_ID,
                            },
                            timeout=120,
                        )

                        response.raise_for_status()
                        result = response.json()

                        answer = result.get(
                            "answer",
                            "No answer was returned.",
                        )

                        case_id = result.get("case_id")
                        pending_review = (
                            result.get("outcome") == "pending_review"
                        )

                        st.write(answer)

                        if pending_review:
                            st.warning(
                                "This case requires human review."
                            )

                        if case_id:
                            st.caption(f"Case ID: {case_id}")

                        # Save the answer and metadata.
                        st.session_state.member_messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "case_id": case_id,
                                "pending_review": pending_review,
                            }
                        )

                    except requests.exceptions.HTTPError:
                        status = response.status_code

                        if status == 401:
                            st.error(
                                "Your JWT is invalid or expired. "
                                "Generate a new member token."
                            )
                        elif status == 403:
                            st.error(
                                "Access denied. Check that you "
                                "are using the correct member JWT."
                            )
                        else:
                            st.error(
                                f"CareAgents returned HTTP {status}."
                            )

                    except requests.exceptions.RequestException:
                        st.error(
                            "Unable to connect to CareAgents. "
                            "Check that FastAPI is running "
                            "on port 8000."
                        )

    # Clear the current chat history.
    if st.session_state.member_messages:
        if st.button("Clear Chat", key="clear_member_chat"):
            st.session_state.member_messages = []
            st.rerun()

    # Case tracking section.
    st.divider()
    st.subheader("📂 Track Your Case")

    st.write(
        "Enter a Case ID to retrieve the latest stored "
        "response and review status."
    )

    case_id_input = st.text_input(
        "Case ID",
        key="member_case_id",
        placeholder="Enter your CareAgents case ID",
    )

    if st.button(
        "Check Case Status",
        type="primary",
        key="check_member_case",
    ):
        if not token:
            st.warning("Enter your Member JWT first.")

        elif not case_id_input.strip():
            st.warning("Please enter a Case ID.")

        else:
            with st.spinner("Retrieving your case..."):
                try:
                    response = requests.get(
                        f"{API_URL}/v1/cases/{case_id_input.strip()}",
                        headers={
                            "Authorization": f"Bearer {token}",
                        },
                        timeout=30,
                    )

                    response.raise_for_status()
                    case = response.json()

                    st.success("Case retrieved successfully!")

                    st.write(
                        "**Case ID:**",
                        case.get("case_id", case_id_input.strip()),
                    )

                    outcome = case.get("outcome", "Unknown")

                    st.write("**Outcome:**", outcome)

                    if outcome == "pending_review":
                        st.warning(
                            "Your case is awaiting human review."
                        )

                    elif outcome == "answered":
                        st.success(
                            "Your case has a recorded answer."
                        )

                    st.markdown("**Response:**")
                    st.write(
                        case.get(
                            "answer",
                            "No response is available yet.",
                        )
                    )

                    if case.get("review_reason"):
                        st.info(
                            "Review reason: "
                            f"{case['review_reason']}"
                        )

                    if "human_reviewed" in case.get("flags", []):
                        st.caption(
                            "This case has been reviewed by staff."
                        )

                except requests.exceptions.HTTPError:
                    status = response.status_code

                    if status == 404:
                        st.error(
                            "Case not found or not accessible "
                            "to this member."
                        )
                    elif status == 401:
                        st.error(
                            "Your Member JWT is invalid or expired."
                        )
                    elif status == 403:
                        st.error(
                            "You do not have permission "
                            "to view this case."
                        )
                    else:
                        st.error(
                            f"Unable to retrieve case. HTTP {status}."
                        )

                except requests.exceptions.RequestException:
                    st.error(
                        "Could not connect to the FastAPI backend."
                    )
