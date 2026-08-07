#ifndef NEXUS_COMMON_LOGGER_HPP
#define NEXUS_COMMON_LOGGER_HPP

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace nexus {

enum class LogLevel {
    INFO,
    WARNING,
    ERROR,
    FATAL,
    DEBUG
};

class Logger {
public:
    static void log(LogLevel level, const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << ms.count()
           << " [" << getLevelString(level) << "] " << message << "\n";

        if (level == LogLevel::ERROR || level == LogLevel::FATAL) {
            std::cerr << ss.str();
        } else {
            std::cout << ss.str();
        }
    }

private:
    static const char* getLevelString(LogLevel level) {
        switch (level) {
            case LogLevel::INFO:    return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR:   return "ERROR";
            case LogLevel::FATAL:   return "FATAL";
            case LogLevel::DEBUG:   return "DEBUG";
            default:                return "UNKNOWN";
        }
    }
};

#define NEXUS_LOG_INFO(msg) ::nexus::Logger::log(::nexus::LogLevel::INFO, msg)
#define NEXUS_LOG_WARN(msg) ::nexus::Logger::log(::nexus::LogLevel::WARNING, msg)
#define NEXUS_LOG_ERROR(msg) ::nexus::Logger::log(::nexus::LogLevel::ERROR, msg)
#define NEXUS_LOG_DEBUG(msg) ::nexus::Logger::log(::nexus::LogLevel::DEBUG, msg)

} // namespace nexus

#endif // NEXUS_COMMON_LOGGER_HPP
