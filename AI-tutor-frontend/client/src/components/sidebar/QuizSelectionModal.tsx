import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Brain, ChevronRight, Clock, Target } from "lucide-react";
import axios from "axios";

interface QuizSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onQuizGenerated: (quizData: any) => void; // ADD THIS LINE
}

interface Week {
  id: string;
  title: string;
  topics: string[];
  file_count: number;
}

export const QuizSelectionModal: React.FC<QuizSelectionModalProps> = ({ isOpen, onClose, onQuizGenerated }) => {
  const [weeks, setWeeks] = React.useState<Week[]>([]);
  const [loadingWeeks, setLoadingWeeks] = React.useState(false);
  const [generatingQuiz, setGeneratingQuiz] = React.useState<string | null>(null); // Track which week is generating
  const [error, setError] = React.useState<string | null>(null);

  // Fetch weeks when modal opens
  React.useEffect(() => {
    if (isOpen) {
      fetchWeeks();
    }
  }, [isOpen]);

  const fetchWeeks = async () => {
    try {
      setLoadingWeeks(true);
      setError(null);
      const response = await axios.get('http://a100-f-01.ai.deakin.edu.au:8000/quiz/weeks');
      setWeeks(response.data.weeks);
    } catch (err) {
      console.error('Failed to fetch weeks:', err);
      setError('Failed to load course weeks. Please try again.');
    } finally {
      setLoadingWeeks(false);
    }
  };

  const handleStartQuiz = async (week: Week) => {
    try {
      setGeneratingQuiz(week.id);
      
      // Call the quiz generation endpoint
      const response = await axios.post('http://a100-f-01.ai.deakin.edu.au:8000/quiz/generate', null, {
        params: { week_id: week.id }
      });
      
      console.log('Quiz generated successfully:', response.data);
      
      // Pass quiz data to parent and close selection modal
      onQuizGenerated(response.data);
      onClose();
      
    } catch (err) {
      console.error('Failed to generate quiz:', err);
      alert('❌ Failed to generate quiz. Please try again.');
    } finally {
      setGeneratingQuiz(null);
    }
  };

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
            className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[80vh] overflow-y-auto p-6 relative mx-4"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ duration: 0.1 }}
          >
            <button
              onClick={onClose}
              className="absolute top-4 right-6 text-gray-500 hover:text-black text-xl"
              disabled={loadingWeeks || generatingQuiz !== null}
            >
              ✕
            </button>

            <div className="mb-6">
              <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
                <Brain className="h-6 w-6 text-blue-600" />
                Select Quiz Week
              </h2>
              <p className="text-gray-600 text-sm">
                Choose a week to test your knowledge. Each quiz contains 5 questions based on that week's content.
              </p>
            </div>

            {/* Loading weeks */}
            {loadingWeeks && (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <p className="mt-2 text-gray-600">Loading course weeks...</p>
              </div>
            )}

            {/* Error state */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                <p className="text-red-800">{error}</p>
                <button 
                  onClick={fetchWeeks}
                  className="mt-2 text-sm text-red-600 hover:text-red-800 underline"
                >
                  Try again
                </button>
              </div>
            )}

            {/* Weeks list */}
            {!loadingWeeks && !error && weeks.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {weeks.map((week) => (
                  <div key={week.id} className="border rounded-xl p-4 hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between mb-3">
                      <h3 className="font-semibold text-lg text-gray-800">{week.title}</h3>
                      <ChevronRight className="h-5 w-5 text-gray-400 mt-1" />
                    </div>
                    
                    <div className="mb-4">
                      <p className="text-sm text-gray-600 mb-2">Main Topics:</p>
                      <div className="flex flex-wrap gap-2">
                        {week.topics.map((topic, idx) => (
                          <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                            {topic}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4 text-sm text-gray-500">
                        <div className="flex items-center gap-1">
                          <Clock className="h-4 w-4" />
                          ~10 min
                        </div>
                        <div className="flex items-center gap-1">
                          <Target className="h-4 w-4" />
                          5 questions
                        </div>
                      </div>
                      
                      <button
                        onClick={() => handleStartQuiz(week)}
                        disabled={generatingQuiz !== null}
                        className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      >
                        {generatingQuiz === week.id ? (
                          <>
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                            Generating...
                          </>
                        ) : (
                          'Start Quiz'
                        )}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* No weeks found */}
            {!loadingWeeks && !error && weeks.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No course weeks found. Please check your course content.
              </div>
            )}

            <div className="mt-6 text-center">
              <p className="text-xs text-gray-500">
                💡 Tip: Quizzes are generated from your week's learning materials and are different each time!
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};