import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Brain, ChevronRight, Clock, Target } from "lucide-react";

interface QuizSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const QuizSelectionModal: React.FC<QuizSelectionModalProps> = ({ isOpen, onClose }) => {
  // Mock data for now - we'll connect to backend later
  const weeks = [
    { 
      id: "week01", 
      title: "Week 1: Introduction to Reinforcement Learning", 
      topics: ["History", "Basic Concepts", "Agents & Environments"] 
    },
    { 
      id: "week02", 
      title: "Week 2: Psychology & Learning Foundations", 
      topics: ["Behavioral Learning", "Cognitive Science", "Neural Basis"] 
    },
    { 
      id: "week03", 
      title: "Week 3: MDPs & Dynamic Programming", 
      topics: ["Markov Decision Processes", "Value Functions", "Policy Iteration"] 
    },
    { 
      id: "week04", 
      title: "Week 4: Monte Carlo Methods", 
      topics: ["Sampling", "Policy Evaluation", "Control Methods"] 
    },
    { 
      id: "week05", 
      title: "Week 5: Temporal Difference Learning", 
      topics: ["TD Learning", "SARSA", "Q-Learning"] 
    },
    { 
      id: "week06", 
      title: "Week 6: Eligibility Traces & DYNA", 
      topics: ["Eligibility Traces", "TD(λ)", "Dyna Architecture"] 
    },
    { 
      id: "week07", 
      title: "Week 7: Function Approximation", 
      topics: ["Linear Methods", "Neural Networks", "Feature Engineering"] 
    },
    { 
      id: "week08", 
      title: "Week 8: Deep RL & Policy Gradients", 
      topics: ["Deep Q-Networks", "Policy Gradients", "Actor-Critic"] 
    },
    { 
      id: "week09", 
      title: "Week 9: Multi-Agent RL & Advising", 
      topics: ["MARL", "Coordination", "Human-AI Interaction"] 
    },
    { 
      id: "week10", 
      title: "Week 10: Multi-Objective Reinforcement Learning", 
      topics: ["Multiple Objectives", "Pareto Fronts", "Scalarization"] 
    },
  ];

  const handleStartQuiz = (week: any) => {
    alert(`🎯 Starting Quiz for ${week.title}!\n\nNext: Generate questions from this week's content.`);
    onClose();
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
                      className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 text-sm font-medium"
                    >
                      Start Quiz
                    </button>
                  </div>
                </div>
              ))}
            </div>

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