#include "LinearMotorNative.h"

#include <Windows.h>
#include <mutex>
#include <vector>

#include "AXL.h"
#include "AXM.h"

namespace
{
    std::recursive_mutex g_mutex;
    bool g_initialized = false;
    long g_axis_no = 0;

    int ToResult(DWORD code)
    {
        return static_cast<int>(code);
    }

    int EnsureInitialized()
    {
        return g_initialized ? LM_OK : LM_ERR_NOT_INITIALIZED;
    }

    int ReadServoAlarmUnlocked(DWORD* alarmStatus)
    {
        if (alarmStatus == nullptr)
        {
            return LM_ERR_INVALID_ARGUMENT;
        }

        const DWORD code = AxmSignalReadServoAlarm(g_axis_no, alarmStatus);
        return ToResult(code);
    }

    int MoveUnlocked(
        DWORD absRelMode,
        double target,
        double velocity,
        double acceleration,
        double deceleration)
    {
        if (velocity <= 0.0 || acceleration <= 0.0 || deceleration <= 0.0)
        {
            return LM_ERR_INVALID_ARGUMENT;
        }

        DWORD alarm = 0;
        int result = ReadServoAlarmUnlocked(&alarm);
        if (result != LM_OK)
        {
            return result;
        }
        if (alarm != 0)
        {
            return LM_ERR_SERVO_ALARM;
        }

        DWORD code = AxmMotSetAbsRelMode(g_axis_no, absRelMode);
        if (code != AXT_RT_SUCCESS)
        {
            return ToResult(code);
        }

        code = AxmMovePos(g_axis_no, target, velocity, acceleration, deceleration);
        return ToResult(code);
    }
}

int LM_CALL LM_Initialize(const char* motFilePath, long axisNo, long irqNo)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (g_initialized)
    {
        return LM_ERR_ALREADY_INITIALIZED;
    }
    if (motFilePath == nullptr || motFilePath[0] == '\0' || axisNo < 0)
    {
        return LM_ERR_INVALID_ARGUMENT;
    }

    DWORD code = AxlOpen(irqNo);
    if (code != AXT_RT_SUCCESS)
    {
        return ToResult(code);
    }

    DWORD moduleStatus = 0;
    code = AxmInfoIsMotionModule(&moduleStatus);
    if (code != AXT_RT_SUCCESS)
    {
        AxlClose();
        return ToResult(code);
    }
    if (moduleStatus != STATUS_EXIST)
    {
        AxlClose();
        return LM_ERR_NO_MOTION_MODULE;
    }

    long axisCount = 0;
    code = AxmInfoGetAxisCount(&axisCount);
    if (code != AXT_RT_SUCCESS)
    {
        AxlClose();
        return ToResult(code);
    }
    if (axisNo >= axisCount)
    {
        AxlClose();
        return LM_ERR_AXIS_OUT_OF_RANGE;
    }

    // The vendor API accepts a mutable char pointer.
    std::vector<char> mutablePath;
    for (const char* p = motFilePath; *p != '\0'; ++p)
    {
        mutablePath.push_back(*p);
    }
    mutablePath.push_back('\0');

    code = AxmMotLoadParaAll(mutablePath.data());
    if (code != AXT_RT_SUCCESS)
    {
        AxlClose();
        return ToResult(code);
    }

    g_axis_no = axisNo;

    // Symmetric S-curve. Absolute/relative mode is selected for every move.
    code = AxmMotSetProfileMode(g_axis_no, 3);
    if (code != AXT_RT_SUCCESS)
    {
        AxlClose();
        return ToResult(code);
    }

    g_initialized = true;
    return LM_OK;
}

int LM_CALL LM_ServoOn()
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    return ToResult(AxmSignalServoOn(g_axis_no, ENABLE));
}

int LM_CALL LM_ServoOff()
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    return ToResult(AxmSignalServoOn(g_axis_no, DISABLE));
}

int LM_CALL LM_MoveAbsolute(
    double position,
    double velocity,
    double acceleration,
    double deceleration)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    return MoveUnlocked(0, position, velocity, acceleration, deceleration);
}

int LM_CALL LM_MoveRelative(
    double distance,
    double velocity,
    double acceleration,
    double deceleration)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    return MoveUnlocked(1, distance, velocity, acceleration, deceleration);
}

int LM_CALL LM_HomeStart()
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    return ToResult(AxmHomeSetStart(g_axis_no));
}

int LM_CALL LM_GetHomeResult(unsigned long* homeResult)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    if (homeResult == nullptr)
    {
        return LM_ERR_INVALID_ARGUMENT;
    }

    DWORD resultValue = 0;
    const DWORD code = AxmHomeGetResult(g_axis_no, &resultValue);
    *homeResult = static_cast<unsigned long>(resultValue);
    return ToResult(code);
}

int LM_CALL LM_Stop(double deceleration)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    if (deceleration <= 0.0)
    {
        return LM_ERR_INVALID_ARGUMENT;
    }
    return ToResult(AxmMoveStop(g_axis_no, deceleration));
}

int LM_CALL LM_EmergencyStop()
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    return ToResult(AxmMoveEStop(g_axis_no));
}

int LM_CALL LM_WaitUntilStopped(unsigned long timeoutMs)
{
    const ULONGLONG startedAt = GetTickCount64();

    while (true)
    {
        {
            std::lock_guard<std::recursive_mutex> lock(g_mutex);
            const int ready = EnsureInitialized();
            if (ready != LM_OK)
            {
                return ready;
            }

            DWORD alarm = 0;
            int result = ReadServoAlarmUnlocked(&alarm);
            if (result != LM_OK)
            {
                return result;
            }
            if (alarm != 0)
            {
                return LM_ERR_SERVO_ALARM;
            }

            DWORD moving = 0;
            const DWORD code = AxmStatusReadInMotion(g_axis_no, &moving);
            if (code != AXT_RT_SUCCESS)
            {
                return ToResult(code);
            }
            if (moving == 0)
            {
                return LM_OK;
            }
        }

        if (timeoutMs > 0 && GetTickCount64() - startedAt >= timeoutMs)
        {
            return LM_ERR_TIMEOUT;
        }
        Sleep(1);
    }
}

int LM_CALL LM_IsMoving(int* isMoving)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    if (isMoving == nullptr)
    {
        return LM_ERR_INVALID_ARGUMENT;
    }

    DWORD moving = 0;
    const DWORD code = AxmStatusReadInMotion(g_axis_no, &moving);
    *isMoving = (moving != 0) ? 1 : 0;
    return ToResult(code);
}

int LM_CALL LM_GetActualPosition(double* position)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    if (position == nullptr)
    {
        return LM_ERR_INVALID_ARGUMENT;
    }
    return ToResult(AxmStatusGetActPos(g_axis_no, position));
}

int LM_CALL LM_GetCommandPosition(double* position)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    if (position == nullptr)
    {
        return LM_ERR_INVALID_ARGUMENT;
    }
    return ToResult(AxmStatusGetCmdPos(g_axis_no, position));
}

int LM_CALL LM_GetServoAlarm(unsigned long* alarmStatus)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    if (alarmStatus == nullptr)
    {
        return LM_ERR_INVALID_ARGUMENT;
    }

    DWORD alarm = 0;
    const int result = ReadServoAlarmUnlocked(&alarm);
    *alarmStatus = static_cast<unsigned long>(alarm);
    return result;
}

int LM_CALL LM_GetLimits(unsigned long* positiveLimit, unsigned long* negativeLimit)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }
    if (positiveLimit == nullptr || negativeLimit == nullptr)
    {
        return LM_ERR_INVALID_ARGUMENT;
    }

    DWORD positive = 0;
    DWORD negative = 0;
    const DWORD code = AxmSignalReadLimit(g_axis_no, &positive, &negative);
    *positiveLimit = static_cast<unsigned long>(positive);
    *negativeLimit = static_cast<unsigned long>(negative);
    return ToResult(code);
}

int LM_CALL LM_ZeroPositionHere()
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    const int ready = EnsureInitialized();
    if (ready != LM_OK)
    {
        return ready;
    }

    DWORD code = AxmStatusSetActPos(g_axis_no, 0.0);
    if (code != AXT_RT_SUCCESS)
    {
        return ToResult(code);
    }
    code = AxmStatusSetCmdPos(g_axis_no, 0.0);
    return ToResult(code);
}

int LM_CALL LM_Close()
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        return LM_OK;
    }

    // Best-effort stop and servo-off before closing the vendor library.
    AxmMoveEStop(g_axis_no);
    AxmSignalServoOn(g_axis_no, DISABLE);
    AxlClose();

    g_initialized = false;
    return LM_OK;
}
