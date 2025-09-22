import { useState, useEffect } from "react"
import { SendIcon, StopCircleIcon, WandSparklesIcon } from "lucide-react"
import { motion } from 'framer-motion'
import { Textarea } from "../ui/textarea"
import { Button } from "../ui/button"
import { useAdjustHeight } from "@/hooks/use-adjust-height"

// Character limit constants
const CHAR_LIMIT = 500;
const WARN_AT = Math.floor(CHAR_LIMIT * 0.8); // 400 characters
const MESSAGES = {
    warn: "Keep your query focused for a better answer.",
    stop: "Character limit reached!",
};

interface ChatInputProps {
    handleSubmit: (message: string, setMessage: React.Dispatch<React.SetStateAction<string>>, resetHeight: () => void) => Promise<void>
    isStreaming: boolean
    handleStopGeneration: () => void
    inputText: string
    setInputText: React.Dispatch<React.SetStateAction<string>>
}

function ChatInput({ handleSubmit, isStreaming, handleStopGeneration, inputText, setInputText }: ChatInputProps) {
    const [message, setMessage] = useState(inputText)
    const [warningMsg, setWarningMsg] = useState<string>("")
    const { textareaRef, adjustHeight, resetHeight } = useAdjustHeight()
    
    useEffect(() => {
        setMessage(inputText)
        adjustHeight()
    }, [inputText])

    // Helper function to get color class based on message length
    const getColorClass = () => {
        if (message.length >= CHAR_LIMIT) return "border-red-400 bg-red-50"
        if (message.length >= WARN_AT) return "border-orange-400 bg-orange-50"
        return "border-gray-300 bg-gray-100"
    }

    const handleMessageOnChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        let value = e.target.value
        if (value.length > CHAR_LIMIT) {
            value = value.slice(0, CHAR_LIMIT)
        }
        adjustHeight()
        setMessage(value)
        setInputText(value)
    }

    const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
        const paste = e.clipboardData.getData('text')
        const newValue = message + paste
        if (newValue.length > CHAR_LIMIT) {
            e.preventDefault()
            const trimmed = newValue.slice(0, CHAR_LIMIT)
            setMessage(trimmed)
            setInputText(trimmed)
            adjustHeight()
        }
    }

    const handleMessageSubmit = (e: React.MouseEvent<HTMLButtonElement, MouseEvent> | React.KeyboardEvent<HTMLTextAreaElement>) => {
        e.preventDefault()
        handleSubmit(message, setMessage, resetHeight)
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleMessageSubmit(e)
        }
    }

    // Update warning message based on character count
    useEffect(() => {
        if (message.length >= CHAR_LIMIT) {
            setWarningMsg(MESSAGES.stop)
        } else if (message.length >= WARN_AT) {
            setWarningMsg(MESSAGES.warn)
        } else {
            setWarningMsg("")
        }
    }, [message])

    return (
        <div className={`border sm:rounded-md p-2 w-full ${getColorClass()}`}>
            <div className="relative">
                <motion.div
                    initial={{ height: "auto" }}
                    animate={{ height: textareaRef.current?.style.height }}
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                >
                    <Textarea
                        ref={textareaRef}
                        value={message}
                        onChange={handleMessageOnChange}
                        onPaste={handlePaste}
                        onKeyDown={handleKeyDown}
                        disabled={isStreaming}
                        className="pl-8 resize-none border-none bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 overflow-y-auto"
                        placeholder="Ask AI a question or make request"
                    />
                </motion.div>
                <WandSparklesIcon className="absolute top-3 left-2 text-muted-foreground h-4 w-4" />
            </div>

            <div className="flex items-end justify-between">
                <div className="text-xs text-gray-400 ml-2 flex flex-row gap-4 justify-between w-full">
                    <p className={message.length >= CHAR_LIMIT ? "text-red-600 font-semibold" : ""}>
                        {message.length}/{CHAR_LIMIT}
                    </p>
                    {message.length > 0 && message.length < CHAR_LIMIT && (
                        <p className="mr-4">Use <code className="bg-muted">shift + enter</code> for new line</p>
                    )}
                </div>

                {warningMsg && (
                    <div className={`text-xs ml-2 mt-1 ${message.length >= CHAR_LIMIT ? "text-red-600" : "text-orange-600"}`}>
                        {warningMsg}
                    </div>
                )}

                {isStreaming ? (
                    <Button onClick={handleStopGeneration} variant={"default"} size="icon">
                        <StopCircleIcon className="animate-spin" />
                    </Button>
                ) : (
                    <Button
                        onClick={handleMessageSubmit}
                        variant={message.length > 0 && message.length <= CHAR_LIMIT ? "default" : "ghost"}
                        size="icon"
                        disabled={message.length === 0 || message.length > CHAR_LIMIT}
                    >
                        <SendIcon />
                    </Button>
                )}
            </div>
        </div>
    )
}

export default ChatInput