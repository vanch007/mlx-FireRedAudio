import React from "react";
import { AppProvider, useApp } from "./context/AppContext";
import { Navbar } from "./components/layout/Navbar";
import { Sidebar } from "./components/layout/Sidebar";
import { AudioPlayer } from "./components/common/AudioPlayer";
import { OverviewView } from "./components/views/OverviewView";
import { QuickToolsView } from "./components/views/QuickToolsView";
import { ProjectsView } from "./components/views/ProjectsView";
import { AssetsVoicesView } from "./components/views/AssetsVoicesView";
import { JobsView } from "./components/views/JobsView";
import { SettingsView } from "./components/views/SettingsView";

const MainContent: React.FC = () => {
  const { currentTab } = useApp();

  return (
    <main className="flex-1 overflow-y-auto pb-28">
      {currentTab === "overview" && <OverviewView />}
      {currentTab === "tools" && <QuickToolsView />}
      {currentTab === "projects" && <ProjectsView />}
      {currentTab === "assets_voices" && <AssetsVoicesView />}
      {currentTab === "jobs" && <JobsView />}
      {currentTab === "settings" && <SettingsView />}
    </main>
  );
};

export const App: React.FC = () => {
  return (
    <AppProvider>
      <div className="flex flex-col h-screen overflow-hidden bg-slate-950 text-slate-100 selection:bg-sky-500 selection:text-white">
        <Navbar />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <MainContent />
        </div>
        <AudioPlayer />
      </div>
    </AppProvider>
  );
};

export default App;
