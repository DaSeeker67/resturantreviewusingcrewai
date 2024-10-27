import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
from langchain_community.llms import HuggingFaceHub


llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token="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",
    task="text-generation",
)

Review_replier = Agent(
    role="Google Review Responder",
    goal="Provide prompt, thoughtful, and attentive responses to Google reviews, ensuring that every customer feels valued.",
    backstory=(
        "You work as the dedicated review responder for Vrindavan Restaurant, where every customer interaction is an opportunity "
        "to enhance our reputation. Positive reviews should be acknowledged warmly, reinforcing the high points mentioned by the customer. "
        "For critical feedback, respond with empathy, requesting further details if necessary, and reassure the customer of our commitment "
        "to improving their experience. Your goal is to ensure each response is handled swiftly and thoughtfully, maintaining our brand’s friendly, attentive personality."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

support_quality_assurance_agent = Agent(
    role="Customer Feedback Assurance Specialist",
    goal="Ensure the highest standard of quality in customer support interactions by promptly reviewing responses for accuracy and empathy.",
    backstory=(
        "You’re part of Vrindavan Restaurant’s quality assurance team, tasked with overseeing the support team’s responses to customer reviews and inquiries. "
        "Your role is to verify that each reply is accurate, complete, and maintains the restaurant’s warm, professional tone. "
        "This involves double-checking responses to ensure they leave no question unanswered, especially in cases of customer dissatisfaction, "
        "to help the team excel in upholding a high standard of customer care."
        "you can make only 1 delegation"
    ),
    allow_delegation=True,
    verbose=True,
    llm=llm
)

inquiry_resolution = Task(
    description=(
        "{customer} just wrote a comment about our restaurant:\n"
        "{comment}\n\n"
        "{person} from {customer} is the one that reached out. "
        "Make sure to use everything you know to provide the best support possible. "
        "You must strive to provide a complete and accurate response to the customer's comment."
    ),
    expected_output=(
        "An exciting response that addresses the customer's comment comprehensively. "
        "Ensure the reply is complete, leaving no questions unanswered, and maintain a helpful and friendly tone throughout."
    ),
    agent=Review_replier,
)

google_review_reply = Task(
    description=(
        "Craft a friendly, thoughtful response to {customer}'s Google review of our service. "
        "If the review is positive, thank the customer warmly and reinforce their positive experience. "
        "If the review is negative, address any concerns with understanding and empathy. "
        "Politely ask for more details if needed, and assure the customer that we’re committed to improving based on their feedback. "
        "The response should be professional but friendly, in line with our brand’s approachable tone."
    ),
    expected_output=(
        "A response ready to be posted on Google Reviews that addresses the customer’s feedback thoroughly, "
        "acknowledging positive comments or, in the case of negative feedback, conveying our commitment to improvement. "
        "Keep the tone warm and genuine, and make sure it reflects our brand’s friendly and professional personality."
    ),
    agent=support_quality_assurance_agent,
)

crew = Crew(
    agents=[Review_replier, support_quality_assurance_agent],
    tasks=[inquiry_resolution, google_review_reply],
    verbose=True,
)

# List of review inputs
reviews = [
    {"customer": "Ramesh", "person": "Ramesh", "comment": "The food has become the worst. No idea if the chef has been changed or what but it is tasteless. Waiters don't treat the crowd in a good way. The ambience is good. Garden area sitting is really nice but the food and the waiter disappoints you so much that you cannot enjoy the meal or the place."},
    {"customer": "Priya", "person": "Priya", "comment": "Amazing food and lovely atmosphere! The staff was very friendly and we had a great time. Highly recommend the paneer tikka!"},
    {"customer": "Vikram", "person": "Vikram", "comment": "Service was okay but the food quality is just average. Expected more given the price point."},
    {"customer": "Anita", "person": "Anita", "comment": "Absolutely loved the garden seating area. The food was delicious, and the waiters were very attentive. We will definitely be coming back!"},
    {"customer": "Deepak", "person": "Deepak", "comment": "Very disappointed with the service. Had to wait for almost 30 minutes before our order was taken. Food was not up to the mark either."}
]

# Process each review and print the result
for review in reviews:
    result = crew.kickoff(inputs=review)
    print(f"Response for {review['customer']}'s review:\n{result}\n")
