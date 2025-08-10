import { useEffect, useRef, useState } from "react";
import ChatInput from "./components/chat/chat-input";
import ChatMessages from "./components/chat/chat-messages";
import ChatPrompts from "./components/chat/chat-prompts";
import { useScrollBottom } from "./hooks/use-scroll-bottom";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "./components/ui/sidebar";
import { AppSidebar } from "./components/sidebar/app-sidebar";
import { MathJaxContext } from "better-react-mathjax";
import axios from "axios";

export interface Message {
  id: number;
  text: string;
  sender: "user" | "assistant";
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesRef = useRef(messages);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      const r = (Math.random() * 16) | 0,
        v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  const { messagesEndRef, scrollToBottom } = useScrollBottom();
  const [isStreaming, setIsStreaming] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [inputText, setInputText] = useState<string>("");
  const [sessionId] = useState<string>(() => {
    let existingSessionId = sessionStorage.getItem("session_id");
    if (!existingSessionId) {
      existingSessionId = generateUUID();
      sessionStorage.setItem("session_id", existingSessionId);
    }
    return existingSessionId;
  });

  const handleStopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const handlePromptSelect = (prompt: string) => {
    setInputText(prompt);
  };

  const handleSubmit = async (
    message: string,
    setMessage: React.Dispatch<React.SetStateAction<string>>,
    resetHeight: () => void
  ) => {
    if (!message.trim()) return;

    const userMessageId = messages.length + 1;
    const assistantMessageId = userMessageId + 1;

    setMessages((prevMessages) => [
      ...prevMessages,
      { id: userMessageId, text: message, sender: "user" },
      { id: assistantMessageId, text: "", sender: "assistant" }, // Placeholder for assistant
    ]);

    setMessage("");
    resetHeight();

    try {
      setIsStreaming(true);
      abortControllerRef.current = new AbortController();
      const signal = abortControllerRef.current.signal;

      // const res = await axios.post("http://10.164.18.48:8000/query", {
      const res = await axios.post("http://a100-f-01.ai.deakin.edu.au:8000/query", {
        query: message,
        size: 5,
        session_id: sessionId,
      });

      const responseText = res.data?.response;
      if (!responseText || typeof responseText !== "string") {
        throw new Error("Invalid response from server");
      }

      const CHUNK_SIZE = 10;
      const CHUNK_DELAY = 10; // ms

      for (let i = 0; i < responseText.length; i += CHUNK_SIZE) {
        if (signal.aborted) {
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, CHUNK_DELAY));
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, text: responseText.slice(0, i + CHUNK_SIZE) }
              : msg
          )
        );
      }
    } catch (error) {
      console.error("API error:", error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, text: "Something went wrong. Please try again." }
            : msg
        )
      );
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <MathJaxContext config={{
      tex2jax: { inlineMath: [["$", "$"], ["\\(", "\\)"]] }
    }}>
      <SidebarProvider>
        <AppSidebar />
        <SidebarInset>
          <header className="flex h-14 shrink-0 items-center gap-2 sticky">
            <div className="flex items-center gap-2 px-3">
              <SidebarTrigger />
            </div>
            <div>
              <p>SIT796 Reinforcement Learning</p>
            </div>
            <div className="ml-auto px-3">
              <div className="flex items-center gap-2 text-sm">
                <div className="hidden font-medium text-muted-foreground md:inline-block">
                  A2I2
                </div>
              </div>
            </div>
          </header>
          <div className="flex flex-col h-[calc(100vh-3.5rem)]">
            <div className="flex-1 overflow-y-auto">
              {messages.length > 0 ? (
                <ChatMessages messages={messages} isStreaming={isStreaming} />
              ) : (
                <ChatPrompts onPromptSelect={handlePromptSelect} />
              )}
              <div ref={messagesEndRef} />
            </div>
            <div className="w-full mx-auto max-w-md sm:max-w-xl md:max-w-2xl lg:max-w-3xl mb-4">
              <div className="flex flex-col items-center gap-2">
                <ChatInput
                  handleSubmit={handleSubmit}
                  isStreaming={isStreaming}
                  handleStopGeneration={handleStopGeneration}
                  inputText={inputText}
                  setInputText={setInputText}
                />
                <p className="text-muted-foreground text-xs">
                  AI can make mistakes. Check important info.
                </p>
              </div>
            </div>
          </div>
        </SidebarInset>
      </SidebarProvider>
    </MathJaxContext>
  );
}

export default App;
