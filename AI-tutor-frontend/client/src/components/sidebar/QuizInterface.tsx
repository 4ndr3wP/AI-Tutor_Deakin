import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Brain, ChevronLeft, ChevronRight, Clock, CheckCircle, XCircle } from "lucide-react";

interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correct_answer: number;
  explanation: string;
}

interface QuizData {
  week_id: string;
  title: string;
  questions: QuizQuestion[];
}

interface QuizInterfaceProps {
  isOpen: boolean;
  onClose: () => void;
  quizData: QuizData | null;
}

interface UserAnswer {
  questionId: string;
  selectedOption: number;
  isCorrect: boolean;
}

export const QuizInterface: React.FC<QuizInterfaceProps> = ({ isOpen, onClose, quizData }) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = React.useState(0);
  const [userAnswers, setUserAnswers] = React.useState<UserAnswer[]>([]);
  const [showResults, setShowResults] = React.useState(false);
  const [selectedOption, setSelectedOption] = React.useState<number | null>(null);

  // Reset state when quiz opens
  React.useEffect(() => {
    if (isOpen && quizData) {
      setCurrentQuestionIndex(0);
      setUserAnswers([]);
      setShowResults(false);
      setSelectedOption(null);
    }
  }, [isOpen, quizData]);

  if (!quizData) return null;

  const currentQuestion = quizData.questions[currentQuestionIndex];
  const totalQuestions = quizData.questions.length;
  const isLastQuestion = currentQuestionIndex === totalQuestions - 1;

  const handleOptionSelect = (optionIndex: number) => {
    setSelectedOption(optionIndex);
  };

  const handleNext = () => {
    if (selectedOption === null) return;

    // Save the answer
    const newAnswer: UserAnswer = {
      questionId: currentQuestion.id,
      selectedOption: selectedOption,
      isCorrect: selectedOption === currentQuestion.correct_answer
    };

    const updatedAnswers = [...userAnswers, newAnswer];
    setUserAnswers(updatedAnswers);

    if (isLastQuestion) {
      // Show results
      setShowResults(true);
    } else {
      // Go to next question
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setSelectedOption(null);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
      // Restore previous answer if exists
      const previousAnswer = userAnswers[currentQuestionIndex - 1];
      setSelectedOption(previousAnswer?.selectedOption ?? null);
      // Remove the current answer from the array
      setUserAnswers(userAnswers.slice(0, -1));
    }
  };

  const calculateScore = () => {
    const correctAnswers = userAnswers.filter(answer => answer.isCorrect).length;
    return Math.round((correctAnswers / totalQuestions) * 100);
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  const restartQuiz = () => {
    setCurrentQuestionIndex(0);
    setUserAnswers([]);
    setShowResults(false);
    setSelectedOption(null);
  };

  if (showResults) {
    const score = calculateScore();
    return (
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-y-auto p-6 relative mx-4"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
            >
              <div className="text-center mb-6">
                <Brain className="h-12 w-12 mx-auto mb-4 text-blue-600" />
                <h2 className="text-2xl font-bold mb-2">Quiz Complete!</h2>
                <div className={`text-4xl font-bold mb-2 ${getScoreColor(score)}`}>
                  {score}%
                </div>
                <p className="text-gray-600">
                  You got {userAnswers.filter(a => a.isCorrect).length} out of {totalQuestions} questions correct
                </p>
              </div>

              <div className="space-y-4 mb-6">
                {quizData.questions.map((question, index) => {
                  const userAnswer = userAnswers[index];
                  const isCorrect = userAnswer?.isCorrect;
                  
                  return (
                    <div key={question.id} className="border rounded-lg p-4">
                      <div className="flex items-start gap-3 mb-3">
                        {isCorrect ? (
                          <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
                        ) : (
                          <XCircle className="h-5 w-5 text-red-600 mt-0.5" />
                        )}
                        <div className="flex-1">
                          <p className="font-medium text-gray-800 mb-2">
                            {index + 1}. {question.question}
                          </p>
                          <p className="text-sm text-gray-600 mb-2">
                            <strong>Your answer:</strong> {question.options[userAnswer.selectedOption]}
                          </p>
                          {!isCorrect && (
                            <p className="text-sm text-green-600 mb-2">
                              <strong>Correct answer:</strong> {question.options[question.correct_answer]}
                            </p>
                          )}
                          <p className="text-sm text-gray-700 bg-gray-50 p-2 rounded">
                            <strong>Explanation:</strong> {question.explanation}
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="flex gap-3">
                <button
                  onClick={restartQuiz}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Retake Quiz
                </button>
                <button
                  onClick={onClose}
                  className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    );
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            className="bg-white rounded-2xl shadow-2xl max-w-3xl w-full max-h-[80vh] overflow-y-auto p-6 relative mx-4"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
          >
            <button
              onClick={onClose}
              className="absolute top-4 right-6 text-gray-500 hover:text-black text-xl"
            >
              ✕
            </button>

            {/* Header */}
            <div className="mb-6">
              <h2 className="text-xl font-bold mb-2 flex items-center gap-2">
                <Brain className="h-5 w-5 text-blue-600" />
                {quizData.title}
              </h2>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    Question {currentQuestionIndex + 1} of {totalQuestions}
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  Progress: {Math.round(((currentQuestionIndex) / totalQuestions) * 100)}%
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${((currentQuestionIndex) / totalQuestions) * 100}%` }}
                ></div>
              </div>
            </div>

            {/* Question */}
            <div className="mb-8">
              <h3 className="text-lg font-semibold text-gray-800 mb-6">
                {currentQuestion.question}
              </h3>

              <div className="space-y-3">
                {currentQuestion.options.map((option, index) => (
                  <button
                    key={index}
                    onClick={() => handleOptionSelect(index)}
                    className={`w-full text-left p-4 rounded-lg border-2 transition-all duration-200 ${
                      selectedOption === index
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
                        selectedOption === index 
                          ? 'border-blue-500 bg-blue-500' 
                          : 'border-gray-300'
                      }`}>
                        {selectedOption === index && (
                          <div className="w-3 h-3 rounded-full bg-white"></div>
                        )}
                      </div>
                      <span className="font-medium text-gray-700">
                        {String.fromCharCode(65 + index)}.
                      </span>
                      <span className="text-gray-800">{option}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Navigation */}
            <div className="flex items-center justify-between">
              <button
                onClick={handlePrevious}
                disabled={currentQuestionIndex === 0}
                className="flex items-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="h-4 w-4" />
                Previous
              </button>

              <button
                onClick={handleNext}
                disabled={selectedOption === null}
                className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLastQuestion ? 'Finish Quiz' : 'Next'}
                {!isLastQuestion && <ChevronRight className="h-4 w-4" />}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};