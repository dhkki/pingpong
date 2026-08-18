#pragma once

#ifdef LINEAR_MOTOR_NATIVE_EXPORTS
#define LM_API extern "C" __declspec(dllexport)
#else
#define LM_API extern "C" __declspec(dllimport)
#endif

#define LM_CALL __stdcall

// Wrapper-defined error codes. Ajin/AXL vendor error codes are returned unchanged.
enum LinearMotorResult
{
    LM_OK = 0,
    LM_ERR_NOT_INITIALIZED = -1001,
    LM_ERR_ALREADY_INITIALIZED = -1002,
    LM_ERR_INVALID_ARGUMENT = -1003,
    LM_ERR_NO_MOTION_MODULE = -1004,
    LM_ERR_AXIS_OUT_OF_RANGE = -1005,
    LM_ERR_TIMEOUT = -1006,
    LM_ERR_SERVO_ALARM = -1007
};

LM_API int LM_CALL LM_Initialize(const char* motFilePath, long axisNo, long irqNo);
LM_API int LM_CALL LM_ServoOn();
LM_API int LM_CALL LM_ServoOff();

LM_API int LM_CALL LM_MoveAbsolute(
    double position,
    double velocity,
    double acceleration,
    double deceleration
);

LM_API int LM_CALL LM_MoveRelative(
    double distance,
    double velocity,
    double acceleration,
    double deceleration
);

LM_API int LM_CALL LM_HomeStart();
LM_API int LM_CALL LM_GetHomeResult(unsigned long* homeResult);

LM_API int LM_CALL LM_Stop(double deceleration);
LM_API int LM_CALL LM_EmergencyStop();
LM_API int LM_CALL LM_WaitUntilStopped(unsigned long timeoutMs);

LM_API int LM_CALL LM_IsMoving(int* isMoving);
LM_API int LM_CALL LM_GetActualPosition(double* position);
LM_API int LM_CALL LM_GetCommandPosition(double* position);
LM_API int LM_CALL LM_GetServoAlarm(unsigned long* alarmStatus);
LM_API int LM_CALL LM_GetLimits(unsigned long* positiveLimit, unsigned long* negativeLimit);

// Dangerous unless the mechanism is physically at the intended origin.
LM_API int LM_CALL LM_ZeroPositionHere();

LM_API int LM_CALL LM_Close();
