#pragma once

#include <format>
#include <iostream>
#include <memory>
#include <string_view>

namespace aloe {

enum class LogLevel { Trace = 0, Info, Warn, Error, None };

class ILogger {
    LogLevel log_level_ = LogLevel::Trace;

public:
    explicit ILogger( LogLevel log_level = LogLevel::Trace )
        : log_level_( log_level ) {}
    virtual ~ILogger() = default;

    virtual void log( LogLevel level, std::string_view message ) = 0;

    LogLevel get_log_level() const { return log_level_; }
    void set_log_level( LogLevel log_level ) { log_level_ = log_level; }
};

class ConsoleLogger final : public ILogger {
public:
    void log( LogLevel level, std::string_view message ) override {
        constexpr const char* reset = "\033[0m";
        const char* colorCode = level_color( level );

        std::ostream& output = ( level == LogLevel::Error || level == LogLevel::Warn ) ? std::cerr : std::cout;

        output << colorCode << "[" << level_to_string( level ) << "] " << reset << message << '\n';
    }

private:
    constexpr const char* level_to_string( LogLevel level ) {
        switch ( level ) {
            case LogLevel::Trace: return "TRACE";
            case LogLevel::Info: return "INFO";
            case LogLevel::Warn: return "WARN";
            case LogLevel::Error: return "ERROR";
            default: return "";
        }
    }

    constexpr const char* level_color( LogLevel level ) {
        switch ( level ) {
            case LogLevel::Trace: return "\033[90m";// Bright gray
            case LogLevel::Info: return "\033[37m"; // White
            case LogLevel::Warn: return "\033[33m"; // Orange/yellow
            case LogLevel::Error: return "\033[91m";// Bright red
            default: return "\033[0m";              // Reset
        }
    }
};

class MockLogger final : public ILogger {
    struct LogEntry {
        LogLevel level;
        std::string message;
    };

public:
    MockLogger()
        : ILogger( LogLevel::Trace ) {}

    void log( LogLevel level, std::string_view message ) override {
        entries.emplace_back( level, std::string( message ) );
    }

    const std::vector<LogEntry>& get_entries() const { return entries; }

private:
    std::vector<LogEntry> entries;
};


inline std::shared_ptr<ILogger>& get_logger_instance() {
    static std::shared_ptr<ILogger> globalLogger = std::make_shared<ConsoleLogger>();
    return globalLogger;
}

inline ILogger& get_logger() {
    return *get_logger_instance();
}

inline void set_logger( std::shared_ptr<ILogger> logger ) {
    get_logger_instance() = std::move( logger );
}

inline LogLevel get_logger_level() {
    return get_logger().get_log_level();
}

inline void set_logger_level( LogLevel level ) {
    get_logger().set_log_level( level );
}

}// namespace aloe

template<typename... ArgsT>
constexpr void log_write( const aloe::LogLevel logLevel, std::format_string<ArgsT...> fmtStr, ArgsT&&... args ) {
    if ( ::aloe::get_logger_level() <= logLevel ) {
        aloe::get_logger().log( logLevel, std::format( fmtStr, std::forward<ArgsT>( args )... ) );
    }
}
