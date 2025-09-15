import * as React from "react"
import {
  MessageCircleQuestion,
  ThumbsUp
} from "lucide-react"
import { FeedbackModal } from "./FeedbackModal"
import { NavSecondary } from "@/components/sidebar/nav-secondary"
import { TeamSwitcher } from "@/components/sidebar/team-switcher"
import DeakinLogo from "/logo.png"

import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarRail,
} from "@/components/ui/sidebar"
import { HelpModal } from "./HelpModal"

const data = {
  teams: [
    {
      name: "Deakin AI Tutor",
      logo: DeakinLogo,
      isImage: true,
      plan: "Enterprise",
    },
  ],
  navSecondary: [
    {
      title: "Help",
      url: "#",
      icon: MessageCircleQuestion,
      onClick: "help", 
    },
    {
      title: "Feedback for Us",
      url: "#",
      icon: ThumbsUp,
      onClick: "feedback", 
    },
  ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const [isHelpModalOpen, setHelpModalOpen] = React.useState(false)
  const [isFeedbackModalOpen, setFeedbackModalOpen] = React.useState(false);

  const handleNavSecondaryClick = (item: any) => {
    if (item.onClick === "help") {
      setHelpModalOpen(true)
    }
    if (item.onClick === "feedback") {
      setFeedbackModalOpen(true)
    }
  }

  return (
    <>
      <Sidebar className="border-r-0" {...props}>
        <SidebarHeader>
          <TeamSwitcher teams={data.teams} />
        </SidebarHeader>
        <SidebarContent>
          <NavSecondary
            items={data.navSecondary}
            className="mt-auto"
            onItemClick={handleNavSecondaryClick}
          />
        </SidebarContent>
        <SidebarRail />
      </Sidebar>

      <HelpModal isOpen={isHelpModalOpen} onClose={() => setHelpModalOpen(false)} />
      <FeedbackModal isOpen={isFeedbackModalOpen} onClose={() => setFeedbackModalOpen(false)} />
    </>
  )
}
