#!/usr/bin/env python3
"""
Gemini vs Groq Comparison Example

This example shows how the updated Gemini chatbot can be used
as a drop-in replacement for your existing Groq implementation.
"""

import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from app.services.chatbot.chatbot_factory import ChatbotFactory

def create_sales_bot_with_gemini():
    """Create a sales bot using Gemini (similar to your existing Groq implementation)"""
    
    # Create Gemini chatbot
    chatbot = ChatbotFactory.create_chatbot("gemini")
    
    # Set custom system prompt for sales
    sales_prompt = """You are a helpful sales assistant. Your role is to:
    1. Greet customers warmly
    2. Understand their needs
    3. Provide relevant product recommendations
    4. Be professional but friendly
    5. Always end with a question to continue the conversation"""
    
    # Create conversation memory
    memory = ConversationBufferWindowMemory(k=4)
    
    return chatbot, memory, sales_prompt

def simulate_sales_conversation(chatbot, memory, system_prompt):
    """Simulate a sales conversation"""
    
    print("🏪 Sales Bot Conversation Started")
    print("=" * 50)
    
    # Initial greeting (similar to your first_question method)
    print("🤖 Bot: Hello! Welcome to our store. I'm here to help you find exactly what you need.")
    print("     How can I assist you today?")
    
    # Simulate customer responses
    customer_inputs = [
        "I'm looking for a laptop for my daughter who's starting college",
        "She'll be studying computer science and needs something powerful but not too expensive",
        "What about battery life? She'll be on campus all day",
        "That sounds perfect! What's the warranty like?"
    ]
    
    for i, customer_input in enumerate(customer_inputs, 1):
        print(f"\n👤 Customer {i}: {customer_input}")
        
        # Get bot response (same as your ask_question method)
        response = chatbot.response(customer_input, memory, system_prompt)
        print(f"🤖 Bot: {response}")
        
        # Add to memory (this happens automatically in the response method)
        print(f"📝 Memory: {len(memory.buffer_as_messages)} messages stored")

def compare_implementations():
    """Compare how Gemini can replace Groq in your existing structure"""
    
    print("🔄 Implementation Comparison")
    print("=" * 50)
    
    # Your existing Groq structure (from llm.py)
    print("📋 Your Existing Groq Structure:")
    print("""
    class SalesBot:
        def __init__(self, api_key, tts):
            self.api_key = api_key
            self.system_prompt = system_prompts["sales"]
            self.groq_client = Groq(api_key=self.api_key)
            self.speaker = tts
            self.first_question_attempted = False
            self.conversation_history = []
        
        def call_llm(self, messages):
            completion = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                temperature=0.7,
            )
            return completion.choices[0].message.content
        
        def ask_question(self, user_input):
            self.conversation_history.append({"role": "user", "content": user_input})
            messages = [self.system_prompt] + self.conversation_history[-4:]
            response = self.call_llm(messages)
            reply_text = self.extract_speech(response)
            self.conversation_history.append({"role": "assistant", "content": reply_text})
            if reply_text:
                self.speaker.text_to_speech(reply_text)
    """)
    
    print("\n🆕 New Gemini Structure:")
    print("""
    # Instead of Groq, use Gemini
    chatbot = ChatbotFactory.create_chatbot("gemini")
    
    # Same conversation flow
    response = chatbot.response(user_input, memory, system_prompt)
    
    # Memory management is handled automatically
    # No need to manually track conversation_history
    """)
    
    print("\n✅ Key Benefits of Gemini Integration:")
    print("1. Same interface - drop-in replacement")
    print("2. Automatic memory management")
    print("3. Better model (gemini-2.5-flash-lite)")
    print("4. Consistent with your existing chatbot factory")
    print("5. Easy to switch between providers")

def main():
    """Main function demonstrating Gemini integration"""
    
    # Check if API key is set
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if not api_key:
        print("❌ Error: GOOGLE_AI_API_KEY environment variable not set")
        print("Please set your Google AI API key:")
        print("export GOOGLE_AI_API_KEY='your-api-key-here'")
        return
    
    try:
        print("🚀 Gemini Chatbot Integration Demo")
        print("=" * 60)
        
        # Show implementation comparison
        compare_implementations()
        
        print("\n" + "=" * 60)
        print("🎯 Live Demo: Sales Bot with Gemini")
        print("=" * 60)
        
        # Create and test the sales bot
        chatbot, memory, system_prompt = create_sales_bot_with_gemini()
        
        # Simulate conversation
        simulate_sales_conversation(chatbot, memory, system_prompt)
        
        print("\n" + "=" * 60)
        print("📊 Final Results:")
        print(f"✅ Provider: {chatbot.provider}")
        print(f"✅ Model: {chatbot.model_name}")
        print(f"✅ Memory: {len(memory.buffer_as_messages)} messages")
        print(f"✅ Temperature: {chatbot.temperature}")
        print(f"✅ Max Tokens: {chatbot.max_tokens}")
        
        print("\n🎉 Integration successful! Gemini can now replace Groq in your existing code.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
