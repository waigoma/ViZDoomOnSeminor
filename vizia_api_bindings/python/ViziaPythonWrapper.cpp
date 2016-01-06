#include <boost/python.hpp>
#include "ViziaDoomGamePython.h"
#include "ViziaDefines.h"
#include <exception>



/*C++ code to expose DoomGamePython via python */

PyObject* createExceptionClass(const char* name, PyObject* baseTypeObj = PyExc_Exception)
{
    using std::string;
    namespace bp = boost::python;

    string scopeName = bp::extract<string>(bp::scope().attr("__name__"));
    string qualifiedName0 = scopeName + "." + name;
    char* qualifiedName1 = const_cast<char*>(qualifiedName0.c_str());

    PyObject* typeObj = PyErr_NewException(qualifiedName1, baseTypeObj, 0);
    if(!typeObj) bp::throw_error_already_set();
    bp::scope().attr(name) = bp::handle<>(bp::borrowed(typeObj));
    return typeObj;
}
//they need to be remembered!!!
PyObject* myExceptionTypeObj5 = NULL;
PyObject* myExceptionTypeObj4 = NULL;
PyObject* myExceptionTypeObj3 = NULL;
PyObject* myExceptionTypeObj2 = NULL; 
PyObject* myExceptionTypeObj = NULL; 
void translate(Vizia::Exception const &e)
{
    PyErr_SetString(myExceptionTypeObj, e.what());
}
void translate2(Vizia::Exception const &e)
{
    PyErr_SetString(myExceptionTypeObj2, e.what());
}
void translate3(Vizia::Exception const &e)
{
    PyErr_SetString(myExceptionTypeObj3, e.what());
}
void translate4(Vizia::Exception const &e)
{
    PyErr_SetString(myExceptionTypeObj4, e.what());
}
void translate5(Vizia::Exception const &e)
{
    PyErr_SetString(myExceptionTypeObj5, e.what());
}

BOOST_PYTHON_MODULE(vizia)
{
    using namespace boost::python;
    using namespace Vizia;
    Py_Initialize();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    import_array();
    
    //exceptions
    myExceptionTypeObj = createExceptionClass("doom_unexpected_exit_exception");
   boost::python::register_exception_translator<Vizia::DoomUnexpectedExitException>(&translate);

    myExceptionTypeObj2 = createExceptionClass("doom_is_not_running_exception");
    boost::python::register_exception_translator<Vizia::DoomIsNotRunningException>(&translate2);

    myExceptionTypeObj3 = createExceptionClass("doom_error_exception");
    boost::python::register_exception_translator<Vizia::DoomErrorException>(&translate3);

    myExceptionTypeObj4 = createExceptionClass("shared_memory_exception");
    boost::python::register_exception_translator<Vizia::SharedMemoryException>(&translate4);

    myExceptionTypeObj5 = createExceptionClass("message_queue_exception");
    boost::python::register_exception_translator<Vizia::MessageQueueException>(&translate5);
    
#define ENUM_VAL_2_PYT(v) .value( #v , v )

    enum_<GameMode>("GameMode")
        ENUM_VAL_2_PYT(PLAYER)
        ENUM_VAL_2_PYT(SPECTATOR);

	enum_<ScreenFormat>("ScreenFormat")
        ENUM_VAL_2_PYT(CRCGCB)
        ENUM_VAL_2_PYT(CRCGCBCA)
        ENUM_VAL_2_PYT(RGB24)
        ENUM_VAL_2_PYT(RGBA32)
        ENUM_VAL_2_PYT(ARGB32)
        ENUM_VAL_2_PYT(CBCGCR)
        ENUM_VAL_2_PYT(CBCGCRCA)
        ENUM_VAL_2_PYT(BGR24)
        ENUM_VAL_2_PYT(BGRA32)
        ENUM_VAL_2_PYT(ABGR32)
        ENUM_VAL_2_PYT(GRAY8)
        ENUM_VAL_2_PYT(DOOM_256_COLORS);

	enum_<Button>("Button")
        ENUM_VAL_2_PYT(ATTACK)
        ENUM_VAL_2_PYT(USE)
        ENUM_VAL_2_PYT(JUMP)
        ENUM_VAL_2_PYT(CROUCH)
        ENUM_VAL_2_PYT(TURN180)
        ENUM_VAL_2_PYT(ALTATTACK)
        ENUM_VAL_2_PYT(RELOAD)
        ENUM_VAL_2_PYT(ZOOM)
        ENUM_VAL_2_PYT(SPEED)
        ENUM_VAL_2_PYT(STRAFE)
        ENUM_VAL_2_PYT(MOVE_RIGHT)
        ENUM_VAL_2_PYT(MOVE_LEFT)
        ENUM_VAL_2_PYT(MOVE_BACKWARD)
        ENUM_VAL_2_PYT(MOVE_FORWARD)
        ENUM_VAL_2_PYT(TURN_RIGHT)
        ENUM_VAL_2_PYT(TURN_LEFT)
        ENUM_VAL_2_PYT(LOOK_UP)
        ENUM_VAL_2_PYT(LOOK_DOWN)
        ENUM_VAL_2_PYT(LAND)
        ENUM_VAL_2_PYT(SELECT_WEAPON1)
        ENUM_VAL_2_PYT(SELECT_WEAPON2)
        ENUM_VAL_2_PYT(SELECT_WEAPON3)
        ENUM_VAL_2_PYT(SELECT_WEAPON4)
        ENUM_VAL_2_PYT(SELECT_WEAPON5)
        ENUM_VAL_2_PYT(SELECT_WEAPON6)
        ENUM_VAL_2_PYT(SELECT_WEAPON7)
        ENUM_VAL_2_PYT(SELECT_WEAPON8)
        ENUM_VAL_2_PYT(SELECT_WEAPON9)
        ENUM_VAL_2_PYT(SELECT_WEAPON0)
        ENUM_VAL_2_PYT(SELECT_NEXT_WEAPON)
        ENUM_VAL_2_PYT(SELECT_PREV_WEAPON)
        ENUM_VAL_2_PYT(DROP_SELECTED_WEAPON)
        ENUM_VAL_2_PYT(ACTIVATE_SELECTED_ITEM)
        ENUM_VAL_2_PYT(SELECT_NEXT_ITEM)
        ENUM_VAL_2_PYT(SELECT_PREV_ITEM)
        ENUM_VAL_2_PYT(DROP_SELECTED_ITEM)
        ENUM_VAL_2_PYT(VIEW_PITCH)
        ENUM_VAL_2_PYT(VIEW_ANGLE)
        ENUM_VAL_2_PYT(FORWARD_BACKWARD)
        ENUM_VAL_2_PYT(LEFT_RIGHT)
        ENUM_VAL_2_PYT(UP_DOWN);

    enum_<GameVar>("GameVar")
        ENUM_VAL_2_PYT(KILLCOUNT)
        ENUM_VAL_2_PYT(ITEMCOUNT)
        ENUM_VAL_2_PYT(SECRETCOUNT)
        ENUM_VAL_2_PYT(HEALTH)
        ENUM_VAL_2_PYT(ARMOR)
        ENUM_VAL_2_PYT(ON_GROUND)
        ENUM_VAL_2_PYT(ATTACK_READY)
        ENUM_VAL_2_PYT(ALTATTACK_READY)
        ENUM_VAL_2_PYT(SELECTED_WEAPON)
        ENUM_VAL_2_PYT(SELECTED_WEAPON_AMMO)
        ENUM_VAL_2_PYT(AMMO1)
        ENUM_VAL_2_PYT(AMMO2)
        ENUM_VAL_2_PYT(AMMO3)
        ENUM_VAL_2_PYT(AMMO4)
        ENUM_VAL_2_PYT(AMMO5)
        ENUM_VAL_2_PYT(AMMO6)
        ENUM_VAL_2_PYT(AMMO7)
        ENUM_VAL_2_PYT(AMMO8)
        ENUM_VAL_2_PYT(AMMO9)
        ENUM_VAL_2_PYT(AMMO0)
        ENUM_VAL_2_PYT(WEAPON1)
        ENUM_VAL_2_PYT(WEAPON2)
        ENUM_VAL_2_PYT(WEAPON3)
        ENUM_VAL_2_PYT(WEAPON4)
        ENUM_VAL_2_PYT(WEAPON5)
        ENUM_VAL_2_PYT(WEAPON6)
        ENUM_VAL_2_PYT(WEAPON7)
        ENUM_VAL_2_PYT(WEAPON8)
        ENUM_VAL_2_PYT(WEAPON9)
        ENUM_VAL_2_PYT(WEAPON0)
        ENUM_VAL_2_PYT(USER1)
        ENUM_VAL_2_PYT(USER2)
        ENUM_VAL_2_PYT(USER3)
        ENUM_VAL_2_PYT(USER4)
        ENUM_VAL_2_PYT(USER5)
        ENUM_VAL_2_PYT(USER6)
        ENUM_VAL_2_PYT(USER7)
        ENUM_VAL_2_PYT(USER8)
        ENUM_VAL_2_PYT(USER9)
        ENUM_VAL_2_PYT(USER10)
        ENUM_VAL_2_PYT(USER11)
        ENUM_VAL_2_PYT(USER12)
        ENUM_VAL_2_PYT(USER13)
        ENUM_VAL_2_PYT(USER14)
        ENUM_VAL_2_PYT(USER15)
        ENUM_VAL_2_PYT(USER16)
        ENUM_VAL_2_PYT(USER17)
        ENUM_VAL_2_PYT(USER18)
        ENUM_VAL_2_PYT(USER19)
        ENUM_VAL_2_PYT(USER20)
        ENUM_VAL_2_PYT(USER21)
        ENUM_VAL_2_PYT(USER22)
        ENUM_VAL_2_PYT(USER23)
        ENUM_VAL_2_PYT(USER24)
        ENUM_VAL_2_PYT(USER25)
        ENUM_VAL_2_PYT(USER26)
        ENUM_VAL_2_PYT(USER27)
        ENUM_VAL_2_PYT(USER28)
        ENUM_VAL_2_PYT(USER29)
        ENUM_VAL_2_PYT(USER30);

	def("DoomTics2Ms", DoomTics2Ms);
	def("Ms2DoomTics", Ms2DoomTics);

    class_<DoomGamePython::PythonState>("State", no_init)
        .def_readonly("number", &DoomGamePython::PythonState::number)
        .def_readonly("image_buffer", &DoomGamePython::PythonState::imageBuffer)
        .def_readonly("vars", &DoomGamePython::PythonState::vars);

    class_<DoomGamePython>("DoomGame", init<>())
		.def("init", &DoomGamePython::init)
		.def("load_config", &DoomGamePython::loadConfig)
		.def("close", &DoomGamePython::close)
		.def("new_episode", &DoomGamePython::newEpisode)
		.def("is_episode_finished", &DoomGamePython::isEpisodeFinished)
		.def("is_new_episode", &DoomGamePython::isNewEpisode)
		.def("set_next_action",&DoomGamePython::setNextAction)
		.def("advance_action",&DoomGamePython::advanceAction)
            
		.def("get_state", &DoomGamePython::getState)
    
        .def("get_game_var", &DoomGamePython::getGameVar)

        .def("get_living_reward", &DoomGamePython::getLivingReward)
        .def("set_living_reward", &DoomGamePython::setLivingReward)
        
        .def("get_death_penalty", &DoomGamePython::getDeathPenalty)
        .def("set_death_penalty", &DoomGamePython::setDeathPenalty)
        
        .def("get_last_reward", &DoomGamePython::getLastReward)
        .def("get_summary_reward", &DoomGamePython::getSummaryReward)
        
        .def("get_last_action", &DoomGamePython::getLastAction)
        
		.def("add_state_available_var", &DoomGamePython::addStateAvailableVar)
		.def("clear_state_available_var", &DoomGamePython::clearStateAvailableVars)
        .def("get_state_available_vars_size", &DoomGamePython::getStateAvailableVarsSize)

		.def("add_available_button", &DoomGamePython::addAvailableButton)
		.def("clear_available_button", &DoomGamePython::clearAvailableButtons)
        .def("get_available_buttons_size", &DoomGamePython::getAvailableButtonsSize)
        .def("set_button_max_value", &DoomGamePython::setButtonMaxValue)

        .def("add_custom_game_arg", &DoomGamePython::addCustomGameArg)
        .def("clear_custom_game_args", &DoomGamePython::clearCustomGameArgs)

        .def("send_game_command", &DoomGamePython::sendGameCommand)

        .def("get_game_buffer", &DoomGamePython::getGameScreen)

        .def("get_game_mode", &DoomGamePython::getGameMode)
        .def("set_game_mode", &DoomGamePython::setGameMode)

		.def("set_doom_game_path", &DoomGamePython::setDoomGamePath)
		.def("set_doom_iwad_path", &DoomGamePython::setDoomIwadPath)
		.def("set_doom_file_path", &DoomGamePython::setDoomFilePath)
		.def("set_doom_map", &DoomGamePython::setDoomMap)
		.def("set_doom_skill", &DoomGamePython::setDoomSkill)
		.def("set_doom_config_path", &DoomGamePython::setDoomConfigPath)

        .def("get_seed", &DoomGamePython::getSeed)
        .def("set_seed", &DoomGamePython::setSeed)
		
		.def("set_auto_new_episode", &DoomGamePython::setAutoNewEpisode)
		.def("set_new_episode_on_timeout", &DoomGamePython::setNewEpisodeOnTimeout)
		.def("set_new_episode_on_player_death", &DoomGamePython::setNewEpisodeOnPlayerDeath)
		.def("set_new_episode_on_map_end", &DoomGamePython::setNewEpisodeOnMapEnd)

        .def("get_episode_start_time", &DoomGamePython::getEpisodeStartTime)
        .def("set_episode_start_time", &DoomGamePython::setEpisodeStartTime)
		.def("get_episode_timeout", &DoomGamePython::getEpisodeTimeout)
		.def("set_episode_timeout", &DoomGamePython::setEpisodeTimeout)
		
        .def("set_disabled_console",&DoomGamePython::setDisabledConsole)
        
		.def("set_screen_resolution", &DoomGamePython::setScreenResolution)
		.def("set_screen_width", &DoomGamePython::setScreenWidth)
		.def("set_screen_height", &DoomGamePython::setScreenHeight)
		.def("set_screen_format", &DoomGamePython::setScreenFormat)
		.def("set_render_hud", &DoomGamePython::setRenderHud)
		.def("set_render_weapon", &DoomGamePython::setRenderWeapon)
		.def("set_render_crosshair", &DoomGamePython::setRenderCrosshair)
		.def("set_render_decals", &DoomGamePython::setRenderDecals)
		.def("set_render_particles", &DoomGamePython::setRenderParticles)
		.def("get_screen_width", &DoomGamePython::getScreenWidth)
		.def("get_screen_height", &DoomGamePython::getScreenHeight)
        .def("get_screen_channels", &DoomGamePython::getScreenChannels)
		.def("get_screen_size", &DoomGamePython::getScreenSize)
		.def("get_screen_pitch", &DoomGamePython::getScreenPitch)
		.def("get_screen_format", &DoomGamePython::getScreenFormat)
        .def("set_visible_window", &DoomGamePython::setVisibleWindow);
}