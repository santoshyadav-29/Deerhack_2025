import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
  useNavigate,
} from "react-router-dom";
import HomePage from "./pages/HomePage";
import CameraPage from "./pages/CameraPage";
import TestSocket from "./pages/TestSocket";

function App() {
  return (
    <Router>
      <div className="flex flex-col h-screen bg-gray-900">
        <div className="flex-grow">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/camera" element={<CameraPage />} />
            <Route path="/test" element={<TestSocket />} />
          </Routes>
        </div>
        <BottomAppBar />
      </div>
    </Router>
  );
}

function BottomAppBar() {
  const location = useLocation();
  const navigate = useNavigate();
  const currentPath = location.pathname;

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-gray-800 border-t border-gray-700 flex justify-around py-2">
      <button
        onClick={() => navigate("/")}
        className={`text-white px-4 py-2 rounded-md font-semibold ${
          currentPath === "/" ? "bg-indigo-600" : "hover:bg-indigo-700"
        }`}
      >
        Home
      </button>
      <button
        onClick={() => navigate("/camera")}
        className={`text-white px-4 py-2 rounded-md font-semibold ${
          currentPath === "/camera" ? "bg-indigo-600" : "hover:bg-indigo-700"
        }`}
      >
        Camera
      </button>
    </nav>
  );
}

export default App;
