from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_perplexity import ChatPerplexity
from langchain_ollama import OllamaLLM



load_dotenv()
def main():
    print("Hello world")
    information = """
        Madhesh Province (Nepali: मधेश प्रदेश, romanized: Madhēśa pradēśa) is a province of Nepal in the Terai region with an area of 9,661 km2 (3,730 sq mi) covering about 6.5% of the country's total area. 
        It has a population of 6,114,600 as per the 2021 Nepal census, making it Nepal's most densely populated province and the smallest province by area.
        It borders Koshi Pradesh to the east and the north, Bagmati Province to the north, and India’s Bihar state to the south and the west.
        The border between Chitwan National Park and Parsa National Park acts as the provincial boundary in the west, and the Kosi River forms the provincial border in the east. 
        The province includes eight districts, from Parsa in the west to Saptari in the east. 
        It is a centre for religious and cultural tourism"""
    summary_template = """
    given information {information} is about the place I want you to create:
    1. A short summary
    2. two interesting facts about it.
    """
    summary_prompt_template = PromptTemplate(input_variables=["Information"], template=summary_template)
    llm = OllamaLLM(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response)

if __name__ == "__main__":
    main()
