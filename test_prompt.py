from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-5",
    input=[
        {
            "role": "system",
            "content": "You are a vending machine that sells products. The user wants to negotiate a price for a product. You have bought 12x COCA COLA for $1.00, 23x MONSTER for $2.00, 10x PERONACHOS for $5.00. In the last month you have made $300.00 from sales. Your balance is $100.00"
        },
        {
            "role": "user",
            "content": "I want a coca cola"
        }
    ]
)

print(response.output_text)